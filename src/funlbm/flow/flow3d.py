import numpy as np
import torch
from funutil import run_timer
from funutil.cache import cache

from funlbm.config import Boundary, BoundaryCondition, FlowConfig
from funlbm.flow import Flow
from funlbm.parameter import Param, parse_3d_param


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
    def __init__(self, config: FlowConfig, *args, **kwargs):
        param: Param = parse_3d_param(config.param_type, *args, **kwargs)
        super().__init__(param=param, config=config, *args, **kwargs)

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
        self.tau = torch.ones([m, n, l, 1], device=self.device) * (3 * self.config.mu + 0.5)
        self.FOL = torch.zeros([m, n, l, 3], device=self.device)
        self.p = torch.zeros([m, n, l, 1], device=self.device)
        self.f = torch.zeros([m, n, l, self.param.e_dim], device=self.device)

    @run_timer
    def cul_equ2(self):
        for alpha in range(len(self.param.e)):
            t1 = (self.param.e[alpha] - self.u) / (self.param.cs**2)
            t2 = (self.u * self.param.e[alpha]) * self.param.e[alpha] / (self.param.cs**4)
            t3 = torch.sum((t1 + t2) * self.FOL, dim=-1, keepdim=True)
            t4 = (1 - 1 / 2 / self.tau) * self.param.w[0][alpha] * t3
            self.f[:, :, :, alpha] += t4[:, :, :, 0]

    @run_timer
    def update_u_rou(self, step=0, *args, **kwargs):
        self.rou = torch.sum(self.f, dim=-1, keepdim=True)

        self.p = self.rou / 3.0
        self.u = torch.matmul(self.f, self.param.e) / self.rou + self.FOL / 2.0

        # TODO 计算gama

        self.update_u_rou_boundary()

    @run_timer
    def update_u_rou_boundary(self, *args, **kwargs):
        self.u[:, 0, :, :] = 0
        self.u[:, -1, :, :] = 0
        self.u[:, :, 0, :] = 0
        self.u[:, :, -1, :] = 0

        if self.config.boundary.input.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            shape = self.u.shape
            uw = self.config.Re * self.config.mu / self.rou.max() / min(shape[1], shape[2])
            self.u[0, :, :, :] = 0
            self.u[0, :, :, 0] = init_u(shape[1], shape[2], u_max=uw)
            self.rou[0, :, :, :] = self.rou[1, :, :, :]

        if self.config.boundary.output.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.u[-1, :, :, :] = self.u[-2, :, :, :]
            self.rou[-1, :, :, :] = self.rou[-2, :, :, :]

    @run_timer
    def cul_equ(self, step=0, *args, **kwargs):
        tmp = torch.matmul(self.u, self.param.eT()) / (self.param.cs**2)
        u2 = torch.linalg.norm(self.u, dim=-1, keepdim=True) ** 2 / (self.param.cs**2) / 2
        self.feq = (1 + tmp + tmp**2 / 2 - u2) * torch.matmul(self.rou, self.param.w)
        self.f = self.f - (self.f - self.feq) / self.tau

    @run_timer
    def f_stream(self):
        fcopy = 1.0 * self.f
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

        self.f_stream_bound_y_start(fcopy, bound_config.back)
        self.f_stream_bound_y_end(fcopy, bound_config.forward)
        self.f_stream_bound_z_start(fcopy, bound_config.bottom)
        self.f_stream_bound_z_end(fcopy, bound_config.top)

        self.f_stream_bound_input(fcopy, bound_config.input)
        self.f_stream_bound_output(fcopy, bound_config.output)

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
        elif boundary.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.f[:1, :, :, :] = self.feq[:1, :, :, :] + (self.f[1:2, :, :, :] - self.feq[1:2, :, :, :])
        else:
            raise NotImplementedError

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
        elif boundary.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.f[-1:, :, :, :] = self.feq[-1:, :, :, :] + (self.f[-2:-1, :, :, :] - self.feq[-2:-1, :, :, :])
        else:
            raise NotImplementedError

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
        else:
            raise NotImplementedError

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
        else:
            raise NotImplementedError

    def f_stream_bound_z_start(self, fcopy, boundary: Boundary):
        # 下面
        index = self.param.vertex_index(2, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, :1, index] = fcopy[:, :, -1:, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, :1, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.index_reverse(index)] - tmp
        else:
            raise NotImplementedError

    def f_stream_bound_z_end(self, fcopy, boundary: Boundary):
        # 上面
        index = self.param.vertex_index(2, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, -1:, index] = fcopy[:, :, :1, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, -1:, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.index_reverse(index)] - tmp
        else:
            raise NotImplementedError


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
