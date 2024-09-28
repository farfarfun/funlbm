import math

import numpy as np

from funlbm.config import Boundary, BoundaryCondition
from funlbm.flow import Flow
from funlbm.parameter import Par3D


class Flow3DStream(Flow):

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
            self.f[ie11:ie12, je11:je12, ke11:ke12, k] = fcopy[
                ie21:ie22, je21:je22, ke21:ke22, k
            ]

        self.f_stream_bound(fcopy)
        # print(self.f[:, :, :, :].min())

    def f_stream_bound(self, fcopy):
        bound_config = self.config.boundary
        self.f_stream_bound_x_start(fcopy, bound_config.left)
        self.f_stream_bound_x_end(fcopy, bound_config.right)
        self.f_stream_bound_y_start(fcopy, bound_config.back)
        self.f_stream_bound_y_end(fcopy, bound_config.forward)
        self.f_stream_bound_z_start(fcopy, bound_config.bottom)
        self.f_stream_bound_z_end(fcopy, bound_config.top)

    def f_stream_bound_x_start(self, fcopy, boundary: Boundary):
        index = self.param.bound_index(0, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:1, :, :, index] = fcopy[-1:, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[:1, :, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:1, :, :, index] = (
                fcopy[:1, :, :, self.param.bound_index_map(index)] - tmp
            )

    def f_stream_bound_x_end(self, fcopy, boundary: Boundary):
        # 右边
        index = self.param.bound_index(0, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[-1:, :, :, index] = fcopy[:1, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[-1:, :, :, index] = fcopy[
                -1:, :, :, self.param.bound_index_map(index)
            ]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[-1:, :, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[-1:, :, :, index] = (
                fcopy[-1:, :, :, self.param.bound_index_map(index)] - tmp
            )

    def f_stream_bound_y_start(self, fcopy, boundary: Boundary):
        # 后面
        index = self.param.bound_index(1, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :1, :, index] = fcopy[:, -1:, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :1, :, index] = fcopy[:, :1, :, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[:, :1, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :1, :, index] = (
                fcopy[:, :1, :, self.param.bound_index_map(index)] - tmp
            )

    def f_stream_bound_y_end(self, fcopy, boundary: Boundary):
        # 前面
        index = self.param.bound_index(1, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, -1:, :, index] = fcopy[:, :1, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, -1:, :, index] = fcopy[
                :, -1:, :, self.param.bound_index_map(index)
            ]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[:, -1:, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, -1:, :, index] = (
                fcopy[:, -1:, :, self.param.bound_index_map(index)] - tmp
            )

    def f_stream_bound_z_start(self, fcopy, boundary: Boundary):
        # 下面
        index = self.param.bound_index(2, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, :1, index] = fcopy[
                :, :, -1:, self.param.bound_index_map(index)
            ]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, :1, index] = fcopy[:, :, :1, self.param.bound_index_map(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[:, :, :1, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :, :1, index] = (
                fcopy[:, :, :1, self.param.bound_index_map(index)] - tmp
            )

    def f_stream_bound_z_end(self, fcopy, boundary: Boundary):
        # 上面
        index = self.param.bound_index(2, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:, :, -1:, index] = fcopy[
                :, :, :1, self.param.bound_index_map(index)
            ]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:, :, -1:, index] = fcopy[
                :, :, -1:, self.param.bound_index_map(index)
            ]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * np.matmul(self.rou[:, :, -1:, :], self.param.w[:, index])
            tmp = (
                tmp
                * np.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :, -1:, index] = (
                fcopy[:, :, -1:, self.param.bound_index_map(index)] - tmp
            )


class Flow3D(Flow3DStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        m, n, l = self.config.size
        m, n, l = int(m), int(n), int(l)
        self.x = np.zeros([m, n, l, 3])
        for i in range(m):
            self.x[i, :, :, 0] = i
        for i in range(n):
            self.x[:, i, :, 1] = i
        for i in range(l):
            self.x[:, :, i, 2] = i

        self.u = np.zeros([m, n, l, 3])
        self.rou = np.ones([m, n, l, 1]) * 1.0
        self.tau = np.ones([m, n, l, 1]) * 0.6364

        self.f = np.zeros([m, n, l, 19])
        self.FOL = np.zeros([m, n, l, 3])
        self.p = np.zeros([m, n, l, 1])

    def update_u_rou(self):
        self.rou = np.sum(self.f, axis=-1, keepdims=True)
        self.u = np.matmul(self.f, self.param.e) / self.rou + self.FOL / 2.0

        if self.config.boundary.left.poiseuille is not None:
            shape = self.u.shape
            for i in range(shape[1]):
                for j in range(shape[2]):
                    y = shape[1] // 2 * math.sqrt(2) - (
                        math.sqrt((i - shape[1] // 2) ** 2 + (j - shape[2] // 2) ** 2)
                    )
                    self.u[0, i, j, 0] = (
                        abs(y * (shape[1] * math.sqrt(2) - y)) * 0.00002
                    )
        # TODO 计算gama

    def cul_equ(self, tau=None):
        tau = tau or self.tau
        tmp = np.matmul(self.u, np.transpose(self.param.e)) / (self.param.cs**2)
        u2 = (
            np.linalg.norm(self.u, axis=-1, keepdims=True) ** 2 / (self.param.cs**2) / 2
        )
        feq = 1 + tmp + tmp**2 / 2 - u2
        weight = np.matmul(self.rou, self.param.w)
        feq = feq * weight
        self.f += (feq - self.f) / tau

    def cul_equ2(self):
        for alpha in range(len(self.param.e)):
            t1 = (Par3D.e[alpha] - self.u) / (Par3D.cs**2)
            t2 = (self.u * Par3D.e[alpha]) * Par3D.e[alpha] / (Par3D.cs**4)
            t3 = (1 - 1 / 2 / self.tau) * np.sum(
                (t1 + t2) * self.FOL, axis=-1, keepdims=True
            )
            self.f[:, :, :, alpha] += t3[:, :, :, 0]
