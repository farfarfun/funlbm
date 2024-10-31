import numpy as np
import torch
from funutil import run_timer
from funutil.cache import cache

from funlbm.config import Boundary, BoundaryCondition
from funlbm.flow import Flow


def cul_u(y, z, a, b, size=100):
    n = np.arange(size)
    res = 32 * (-1) ** (n + 1) / (2 * n + 1) ** 3 / np.pi**3
    res = res * np.cos((2 * n + 1) / 2 / a * np.pi * y)
    res = res * np.cosh((2 * n + 1) / 2 / a * np.pi * z) / np.cosh((2 * n + 1) / 2 / a * np.pi * b)

    res = 1 - y**2 / a**2 + np.sum(res)
    return round(res, 6)


@cache
def init_u(a, b, u_max=0.01, n_max=100):
    res = np.zeros([a, b])
    for i in range(a):
        yi = i - (a - 1) / 2.0
        for j in range(b):
            zi = j - (b - 1) / 2.0
            res[i, j] = u_max * cul_u(yi, zi, a / 2.0, b / 2.0, size=n_max)
    return torch.from_numpy(res)


def tran3d(ii, total):
    if ii == 1:
        return 1, total, 0, total - 1
    elif ii == -1:
        return 0, total - 1, 1, total
    else:
        return 0, total, 0, total


class FlowD3(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        m, n, l = self.config.size
        m, n, l = int(m), int(n), int(l)
        self.x = torch.zeros([m, n, l, 3], device=self.device)
        for i in range(m):
            self.x[i, :, :, 0] = i
        for i in range(n):
            self.x[:, i, :, 1] = i
        for i in range(l):
            self.x[:, :, i, 2] = i

        self.u = torch.zeros([m, n, l, 3], device=self.device)
        self.rou = torch.ones([m, n, l, 1], device=self.device) * 1.0
        self.tau = torch.ones([m, n, l, 1], device=self.device) * 0.6364
        self.FOL = torch.zeros([m, n, l, 3], device=self.device)
        self.p = torch.zeros([m, n, l, 1], device=self.device)
        self.f = torch.zeros([m, n, l, self.param.e_dim], device=self.device)

    @run_timer
    def update_u_rou(self):
        self.rou = torch.sum(self.f, axis=-1, keepdims=True)
        self.p = self.rou / 3.0
        self.u = torch.matmul(self.f, self.param.e) / self.rou + self.FOL / 2.0

        if self.config.boundary.input.poiseuille is not None:
            shape = self.u.shape
            self.u[0, :, :, 0] = init_u(shape[1], shape[2], u_max=self.config.boundary.input.get("uw", 0.001))

        # TODO 计算gama

    @run_timer
    def cul_equ(self, tau=None):
        tau = tau or self.tau
        tmp = torch.matmul(self.u, torch.transpose(self.param.e, 0, 1)) / (self.param.cs**2)
        u2 = torch.linalg.norm(self.u, axis=-1, keepdims=True) ** 2 / (self.param.cs**2) / 2
        feq = 1 + tmp + tmp**2 / 2 - u2
        weight = torch.matmul(self.rou, self.param.w)
        feq = feq * weight
        self.f += (feq - self.f) / tau

    @run_timer
    def cul_equ2(self):
        for alpha in range(len(self.param.e)):
            t1 = (self.param.e[alpha] - self.u) / (self.param.cs**2)
            t2 = (self.u * self.param.e[alpha]) * self.param.e[alpha] / (self.param.cs**4)
            t3 = (1 - 1 / 2 / self.tau) * torch.sum((t1 + t2) * self.FOL, axis=-1, keepdims=True)
            self.f[:, :, :, alpha] += t3[:, :, :, 0]

    @run_timer
    def f_stream(self):
        fcopy = 1 * self.f
        i2, j2, k2 = self.f.shape[:3]

        for k, (e1, e2, e3) in enumerate(self.param.e):
            ie11, ie12, ie21, ie22 = tran3d(e1, i2)
            je11, je12, je21, je22 = tran3d(e2, j2)
            ke11, ke12, ke21, ke22 = tran3d(e3, k2)
            self.f[ie11:ie12, je11:je12, ke11:ke12, k] = fcopy[ie21:ie22, je21:je22, ke21:ke22, k]

        self.f_stream_bound(fcopy)

    @run_timer
    def f_stream_bound(self, fcopy):
        bound_config = self.config.boundary
        self.f_stream_bound_input(fcopy, bound_config.input)
        self.f_stream_bound_output(fcopy, bound_config.output)
        self.f_stream_bound_y_start(fcopy, bound_config.back)
        self.f_stream_bound_y_end(fcopy, bound_config.forward)
        self.f_stream_bound_z_start(fcopy, bound_config.bottom)
        self.f_stream_bound_z_end(fcopy, bound_config.top)

    def f_stream_bound_input(self, fcopy, boundary: Boundary):
        index = self.param.vertex_index(0, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:1, :, :, index] = fcopy[-1:, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:1, :, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.index_reverse(index)] - tmp

    def f_stream_bound_output(self, fcopy, boundary: Boundary):
        # 右边
        index = self.param.vertex_index(0, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[-1:, :, :, index] = fcopy[:1, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[-1:, :, :, index] = fcopy[-1:, :, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[-1:, :, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[-1:, :, :, index] = fcopy[-1:, :, :, self.param.index_reverse(index)] - tmp

    def f_stream_bound_y_start(self, fcopy, boundary: Boundary):
        # 后面
        index = self.param.vertex_index(1, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :1, :, index] = fcopy[:, -1:, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :1, :, index] = fcopy[:, :1, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :1, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :1, :, index] = fcopy[:, :1, :, self.param.index_reverse(index)] - tmp

    def f_stream_bound_y_end(self, fcopy, boundary: Boundary):
        # 前面
        index = self.param.vertex_index(1, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, -1:, :, index] = fcopy[:, :1, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, -1:, :, index] = fcopy[:, -1:, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, -1:, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, -1:, :, index] = fcopy[:, -1:, :, self.param.index_reverse(index)] - tmp

    def f_stream_bound_z_start(self, fcopy, boundary: Boundary):
        # 下面
        index = self.param.vertex_index(2, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, :1, index] = fcopy[:, :, -1:, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, :1, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.index_reverse(index)] - tmp

    def f_stream_bound_z_end(self, fcopy, boundary: Boundary):
        # 上面
        index = self.param.vertex_index(2, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, -1:, index] = fcopy[:, :, :1, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, -1:, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.index_reverse(index)] - tmp


class FlowD3Q27(FlowD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlowD3Q19(FlowD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlowD3Q15(FlowD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlowD3Q13(FlowD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
