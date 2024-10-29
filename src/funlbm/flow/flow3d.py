import torch
from funutil import run_timer

from funlbm.config import Boundary, BoundaryCondition
from funlbm.flow import Flow
from funlbm.flow.init import init_u


class Flow3DStream(Flow):
    @run_timer
    def f_stream(self):
        fcopy = 1 * self.f
        # fcopy = np.ones(self.f.shape)
        i2, j2, k2 = self.f.shape[:3]

        def tran(ii, total):
            if ii == 1:
                return 1, total, 0, total - 1
            elif ii == -1:
                return 0, total - 1, 1, total
            else:
                return 0, total, 0, total

        for k, (e1, e2, e3) in enumerate(self.param.e):
            ie11, ie12, ie21, ie22 = tran(e1, i2)
            je11, je12, je21, je22 = tran(e2, j2)
            ke11, ke12, ke21, ke22 = tran(e3, k2)
            # print(k, '|', e1, e2, e3,
            #       '|', ie11, ie12, ie21, ie22,
            #       '|', je11, je12, je21, je22,
            #       '|', ke11, ke12, ke21, ke22)
            self.f[ie11:ie12, je11:je12, ke11:ke12, k] = fcopy[ie21:ie22, je21:je22, ke21:ke22, k]

        self.f_stream_bound(fcopy)
        # print(self.f[:, :, :, :].min())

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
        index = self.param.bound_index(0, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:1, :, :, index] = fcopy[-1:, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:1, :, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.bound_index_map(index)] - tmp

    def f_stream_bound_output(self, fcopy, boundary: Boundary):
        # 右边
        index = self.param.bound_index(0, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[-1:, :, :, index] = fcopy[:1, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[-1:, :, :, index] = fcopy[-1:, :, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[-1:, :, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[-1:, :, :, index] = fcopy[-1:, :, :, self.param.bound_index_map(index)] - tmp

    def f_stream_bound_y_start(self, fcopy, boundary: Boundary):
        # 后面
        index = self.param.bound_index(1, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :1, :, index] = fcopy[:, -1:, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :1, :, index] = fcopy[:, :1, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :1, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :1, :, index] = fcopy[:, :1, :, self.param.bound_index_map(index)] - tmp

    def f_stream_bound_y_end(self, fcopy, boundary: Boundary):
        # 前面
        index = self.param.bound_index(1, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, -1:, :, index] = fcopy[:, :1, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, -1:, :, index] = fcopy[:, -1:, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, -1:, :, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, -1:, :, index] = fcopy[:, -1:, :, self.param.bound_index_map(index)] - tmp

    def f_stream_bound_z_start(self, fcopy, boundary: Boundary):
        # 下面
        index = self.param.bound_index(2, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, :1, index] = fcopy[:, :, -1:, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, :1, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.bound_index_map(index)] - tmp

    def f_stream_bound_z_end(self, fcopy, boundary: Boundary):
        # 上面
        index = self.param.bound_index(2, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, -1:, index] = fcopy[:, :, :1, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:, :, -1:, :], self.param.w[:, index])
            tmp = tmp * torch.matmul(self.param.e[index], boundary.get("uw")) / self.param.cs**2
            self.f[:, :, -1:, index] = fcopy[:, :, -1:, self.param.bound_index_map(index)] - tmp


class Flow3D(Flow3DStream):
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

        self.f = torch.zeros([m, n, l, 19], device=self.device)
        self.FOL = torch.zeros([m, n, l, 3], device=self.device)
        self.p = torch.zeros([m, n, l, 1], device=self.device)

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
        tmp = torch.matmul(self.u, torch.transpose(self.param.e, 0, 1))
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
