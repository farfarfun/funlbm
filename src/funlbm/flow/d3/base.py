import numpy as np
import torch
from farcache import cache
from funutil import run_timer

from funlbm.config import Boundary, BoundaryCondition
from funlbm.flow import FlowBase, FlowConfig


def cul_u(y: float, z: float, a: float, b: float, size: int = 100) -> float:
    """按矩形管道层流解析解计算 (y, z) 处的无量纲轴向速度分量。

    使用傅里叶级数展开近似矩形截面管道层流的解析速度分布，
    用于构造 `init_u` 的入口/出口非平衡速度边界条件。

    Args:
        y: 截面内第一个方向的坐标，坐标原点在截面中心。
        z: 截面内第二个方向的坐标，坐标原点在截面中心。
        a: 截面在 y 方向的半宽。
        b: 截面在 z 方向的半宽。
        size: 级数展开的项数，越大越精确但计算越慢。

    Returns:
        (y, z) 处的无量纲速度，保留 6 位小数。
    """
    # 预计算常量以提高性能
    n = np.arange(size)
    pi = np.pi
    n_terms = 2 * n + 1

    # 分解计算步骤提高可读性
    base_term = 32 * (-1) ** (n + 1) / (n_terms**3 * pi**3)
    y_term = np.cos(n_terms * pi * y / (2 * a))
    z_term = np.cosh(n_terms * pi * z / (2 * a)) / np.cosh(n_terms * pi * b / (2 * a))

    # 计算级数和
    series_sum = np.sum(base_term * y_term * z_term)
    result = 1 - (y / a) ** 2 + series_sum

    return round(result, 6)


@cache
def init_u(a: int, b: int, u_max: float = 0.01, n_max: int = 100) -> torch.Tensor:
    """生成矩形截面 (a x b) 上的层流速度分布，用于非平衡边界初始化。

    结果带缓存（`farcache`），相同 (a, b, u_max, n_max) 组合只计算一次。

    Args:
        a: 截面在第一个方向上的网格点数。
        b: 截面在第二个方向上的网格点数。
        u_max: 截面中心的最大速度。
        n_max: 传给 `cul_u` 的级数展开项数。

    Returns:
        形状为 (a, b) 的速度分布张量。
    """
    # 使用 meshgrid 创建坐标网格
    y = np.linspace(-(a - 1) / 2, (a - 1) / 2, a)
    z = np.linspace(-(b - 1) / 2, (b - 1) / 2, b)
    Y, Z = np.meshgrid(y, z, indexing="ij")

    # 向量化计算
    res = u_max * np.vectorize(lambda y, z: cul_u(y, z, a / 2.0, b / 2.0, n_max))(Y, Z)
    return torch.from_numpy(res)


def tran3d(direction: int, total: int) -> tuple[int, int, int, int]:
    """计算流场边界索引。

    Args:
        direction: 方向 (-1, 0, 1)
        total: 总长度

    Returns:
        两组起止索引 ``(start1, end1, start2, end2)``。
    """
    if direction == 1:
        return 1, total, 0, total - 1
    elif direction == -1:
        return 0, total - 1, 1, total
    else:
        return 0, total, 0, total


class FlowD3(FlowBase):
    """3D流场类

    实现了3D流场的基本功能,包括:
    - 流场初始化
    - 边界条件处理
    - 流场演化计算

    Args:
        config: 流场配置对象
    """

    def __init__(self, config: FlowConfig, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

    def init(self, *args: object, **kwargs: object) -> None:
        """初始化3D流场

        初始化网格坐标和物理量,包括:
        - 坐标场
        - 速度场
        - 密度场
        - 分布函数等
        """
        m, n, depth = self.config.size
        m, n, depth = int(m + 1), int(n + 1), int(depth + 1)

        # 使用 arange 和 meshgrid 替代循环来初始化坐标
        x = torch.arange(m, device=self.device)
        y = torch.arange(n, device=self.device)
        z = torch.arange(depth, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        self.x = torch.stack([X, Y, Z], dim=-1)

        # 初始化物理量
        shape = (m, n, depth)
        self.u = torch.zeros([*shape, 3], device=self.device)
        self.rou = torch.ones([*shape, 1], device=self.device)
        self.tau = torch.full_like(self.rou, 3 * self.config.mu + 0.5)
        self.FOL = torch.zeros_like(self.u)
        self.p = torch.zeros_like(self.rou)
        self.f = torch.zeros([*shape, self.param.e_dim], device=self.device)
        self.f = torch.matmul(self.rou, self.param.w)

        self.update_u_rou()

    @run_timer
    def cul_equ2(self) -> None:
        """计算并叠加浸没边界力导致的第二平衡态修正。

        Returns:
            无返回值，计算结果直接写入 ``self.f``。
        """
        # 预计算一些常量
        cs2 = self.param.cs**2
        cs4 = cs2**2

        for alpha in range(len(self.param.e)):
            t1 = (self.param.e[alpha] - self.u) / cs2
            t2 = ((self.u * self.param.e[alpha]) * self.param.e[alpha]) / cs4
            t3 = torch.sum((t1 + t2) * self.FOL, dim=-1, keepdim=True)
            t4 = (1 - 1 / (2 * self.tau)) * self.param.w[0][alpha] * t3
            self.f[:, :, :, alpha] += t4[:, :, :, 0]

    @run_timer
    def update_u_rou(self, *args: object, **kwargs: object) -> None:
        """由分布函数更新密度、压力和速度场。

        Args:
            *args: 预留的扩展位置参数。
            **kwargs: 预留的扩展关键字参数。

        Returns:
            无返回值，结果写入流场属性。
        """
        self.rou = torch.sum(self.f, dim=-1, keepdim=True)
        self.p = self.rou / 3.0
        self.u = (torch.matmul(self.f, self.param.e) + self.FOL / 2.0) / self.rou
        # TODO 计算gama
        self.update_u_rou_boundary()

    @run_timer
    def update_u_rou_boundary(self, *args, **kwargs):
        self.u[0, :, :, :] = 0
        self.u[-1, :, :, :] = 0
        self.u[:, 0, :, :] = 0
        self.u[:, -1, :, :] = 0
        self.u[:, :, 0, :] = 0
        self.u[:, :, -1, :] = 0

        if self.config.boundary.input.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            shape = self.u.shape
            uw = (
                self.config.Re
                * self.config.mu
                / self.rou.max()
                / min(shape[1], shape[2])
            )
            self.u[0, :, :, 0] = init_u(shape[1], shape[2], u_max=uw)
            self.rou[0, :, :, :] = self.rou[1, :, :, :]

        if self.config.boundary.output.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.rou[-1, :, :, :] = self.rou[-2, :, :, :]
            self.u[-1, :, :, :] = self.u[-2, :, :, :]

    @run_timer
    def cul_equ(self, step: int = 0, *args: object, **kwargs: object) -> None:
        """计算 BGK 碰撞后的平衡态分布函数。

        Args:
            step: 当前模拟步数，保留给扩展实现。
            *args: 预留的扩展位置参数。
            **kwargs: 预留的扩展关键字参数。

        Returns:
            无返回值，计算结果写入 ``self.f`` 和 ``self.feq``。
        """
        tmp = torch.matmul(self.u, self.param.eT()) / (self.param.cs**2)
        u2 = (
            torch.linalg.norm(self.u, dim=-1, keepdim=True) ** 2
            / (self.param.cs**2)
            / 2
        )
        self.feq = (1 + tmp + tmp**2 / 2 - u2) * torch.matmul(self.rou, self.param.w)
        self.f = self.f - (self.f - self.feq) / self.tau

    @run_timer
    def f_stream(self) -> None:
        """沿离散速度方向迁移分布函数，并应用边界条件。

        Returns:
            无返回值，迁移结果写入 ``self.f``。
        """
        fcopy = self.f.clone()

        # 跳过静止粒子(e1=e2=e3=0)
        for k, e in enumerate(self.param.e[1:], 1):  # 从索引1开始
            e1, e2, e3 = e

            # 使用列表推导式简化代码
            shifts_dims = [
                (int(shift), dim)
                for shift, dim in zip([e1, e2, e3], range(3))
                if shift != 0
            ]

            f_temp = fcopy[..., k]
            for shift, dim in shifts_dims:
                f_temp = torch.roll(f_temp, shifts=shift, dims=dim)
            self.f[..., k] = f_temp

        self.f_stream_bound(fcopy)

    @run_timer
    def f_stream_bound(self, fcopy):
        bound_config = self.config.boundary

        # 注意：入口/出口边界的处理必须在其他边界之前，
        # 因为入口/出口边界的处理可能会影响到其他边界
        self.f_stream_bound_x_start(fcopy, bound_config.input)
        self.f_stream_bound_x_end(fcopy, bound_config.output)

        # 其他边界的处理顺序不重要
        self.f_stream_bound_y_start(fcopy, bound_config.back)
        self.f_stream_bound_y_end(fcopy, bound_config.forward)
        self.f_stream_bound_z_start(fcopy, bound_config.bottom)
        self.f_stream_bound_z_end(fcopy, bound_config.top)

    def f_stream_bound_x_start(self, fcopy: torch.Tensor, boundary: Boundary) -> None:
        index = self.param.vertex_index(0, 1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[:1, :, :, index] = fcopy[-1:, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[:1, :, :, index] = fcopy[:1, :, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[:1, :, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:1, :, :, index] = (
                fcopy[:1, :, :, self.param.index_reverse(index)] - tmp
            )
        elif boundary.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.f[:1, :, :, :] = self.feq[:1, :, :, :] + (
                self.f[1:2, :, :, :] - self.feq[1:2, :, :, :]
            )
        else:
            raise NotImplementedError

    def f_stream_bound_x_end(self, fcopy: torch.Tensor, boundary: Boundary) -> None:
        # 右边
        index = self.param.vertex_index(0, -1)
        if boundary.is_condition(BoundaryCondition.PERIODICAL):
            self.f[-1:, :, :, index] = fcopy[:1, :, :, index]
        elif boundary.is_condition(BoundaryCondition.WALL):
            self.f[-1:, :, :, index] = fcopy[-1:, :, :, self.param.index_reverse(index)]
        elif boundary.is_condition(BoundaryCondition.WALL_WITH_SPEED):
            tmp = 2 * torch.matmul(self.rou[-1:, :, :, :], self.param.w[:, index])
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[-1:, :, :, index] = (
                fcopy[-1:, :, :, self.param.index_reverse(index)] - tmp
            )
        elif boundary.is_condition(BoundaryCondition.NON_EQUILIBRIUM):
            self.f[-1:, :, :, :] = self.feq[-1:, :, :, :] + (
                self.f[-2:-1, :, :, :] - self.feq[-2:-1, :, :, :]
            )
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
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :1, :, index] = (
                fcopy[:, :1, :, self.param.index_reverse(index)] - tmp
            )
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
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, -1:, :, index] = (
                fcopy[:, -1:, :, self.param.index_reverse(index)] - tmp
            )
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
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :, :1, index] = (
                fcopy[:, :, :1, self.param.index_reverse(index)] - tmp
            )
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
            tmp = (
                tmp
                * torch.matmul(self.param.e[index], boundary.get("uw"))
                / self.param.cs**2
            )
            self.f[:, :, -1:, index] = (
                fcopy[:, :, -1:, self.param.index_reverse(index)] - tmp
            )
        else:
            raise NotImplementedError
