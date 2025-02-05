import math

import numpy as np
import torch
from funutil import deep_get, run_timer

from funlbm.base import Worker
from funlbm.config.base import BaseConfig
from funlbm.particle.coord import CoordConfig, Coordinate
from funlbm.util import logger


class ParticleConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_config: CoordConfig = CoordConfig()

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.coord_config.from_json(deep_get(config_json, "coord"))


def cul_point(xl, yl, zl, xr, yr, zr, cul_value, dx=0.5, device="cuda"):
    # 直接使用torch计算并在GPU上运行
    x = torch.linspace(xl, xr, int((xr - xl) / dx), device=device)
    y = torch.linspace(yl, yr, int((yr - yl) / dx), device=device)
    z = torch.linspace(zl, zr, int((zr - zl) / dx), device=device)

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    value = cul_value(X, Y, Z)

    # 确保使用浮点类型进行计算
    inner = (value < 0).to(dtype=torch.float32)
    outer = (value > 0).to(dtype=torch.float32) * 2
    S = inner + outer

    T = S[:-1, :-1, :-1]
    T = T + S[1:, :-1, :-1] + S[:-1, 1:, :-1] + S[:-1, :-1, 1:]
    T = T + S[:-1, 1:, 1:] + S[1:, :-1, 1:] + S[1:, 1:, :-1]
    T = T + S[1:, 1:, 1:]

    # 计算平均值
    xm = T.mean(dim=2).mean(dim=1)  # 使用dim替代axis
    ym = T.mean(dim=2).mean(dim=0)
    zm = T.mean(dim=0).mean(dim=0)

    result = []
    for i in range(T.shape[0]):
        if xm[i] == 16:
            continue
        for j in range(T.shape[1]):
            if ym[j] == 16:
                continue
            for k in range(T.shape[2]):
                if zm[k] == 16:
                    continue
                if T[i, j, k] == 16:
                    continue

                result.append([X[i][j][k].item(), Y[i][j][k].item(), Z[i][j][k].item()])

    logger.info(f"lagrange size:{len(result)}")
    return np.array(result, dtype=np.float32)


class Particle(Worker):
    """
    粒子基类

    属性:
        config (ParticleConfig): 粒子配置
        coord (Coordinate): 坐标系对象
        mass (float): 粒子质量
        area (float): 粒子表面积
        I (Tensor): 惯性矩
        rou (float): 粒子密度
        angle (Tensor): 粒子方向
        cx (Tensor): 质心坐标 [x,y,z]
        cr (Tensor): 质心半径 [a,b,b]
        cu (Tensor): 质心速度 [vx,vy,vz]
        cw (Tensor): 质心角速度 [wx,wy,wz]
        cF (Tensor): 质心合外力
        cT (Tensor): 质心合外力矩
        lx (Tensor): 拉格朗日点坐标 [m,i,3]
        lF (Tensor): 拉格朗日点受力 [m,i,3]
        lm (Tensor): 拉格朗日点质量
        lu (Tensor): 拉格朗日点速度 [m,i,3]
        lrou (Tensor): 拉格朗日点密度
    """

    def __init__(self, config: ParticleConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: ParticleConfig = config or ParticleConfig()
        self.coord: Coordinate = Coordinate(
            config=self.config.coord_config, *args, **kwargs
        )

        # 颗粒质量[1]
        self.mass = None
        # 颗粒面积[1]
        self.area = None
        # 惯性矩
        self.I = None
        # 颗粒密度[1]
        self.rou = None
        # 颗粒方向[i,j,k]
        self.angle = None

        # 质心坐标[i,j,k]
        self.cx = torch.tensor(
            self.config.coord_config.center, device=self.device, dtype=torch.float32
        )
        # 质心半径[a,b,b]
        self.cr = 5 * torch.ones(5, device=self.device, dtype=torch.float32)
        # 质心速度[i,j,k]
        self.cu = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心角速度[i,j,k]
        self.cw = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cF = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cT = torch.zeros(3, device=self.device, dtype=torch.float32)

        self._lagrange: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )
        # 拉格朗日点的坐标[m,i,3]
        self.lx: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )
        # 拉格朗日点上的力[m,i,3]
        self.lF: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )
        # 拉格朗日点的质量
        self.lm: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )
        # 拉格朗日点速度[m,i,3]
        self.lu: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )
        # 拉格朗日点速度[m,i,3]
        self.lrou: torch.Tensor = torch.zeros(
            [0], device=self.device, dtype=torch.float32
        )

    def _init(self, dx=1, *args, **kwargs):
        raise NotImplementedError("还没实现")

    def init(self, *args, **kwargs):
        self.rou = float(self.config.get("rou", 1.0))
        self._init(*args, **kwargs)
        shape = self._lagrange.shape
        self.lx = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lF = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lu = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lm = torch.full(
            (shape[0], 1), self.area / shape[0], device=self.device, dtype=torch.float32
        )
        self.lrou = torch.empty((shape[0], 1), device=self.device, dtype=torch.float32)

    @run_timer
    def update_from_lar(self, dt: float, gl: float = 9.8, rouf: float = 1.0) -> None:
        """
        从拉格朗日坐标更新粒子状态。

        参数:
            dt: 时间步长
            gl: 重力加速度
            rouf: 流体密度

        异常:
            ValueError: 当输入参数无效时抛出
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if rouf <= 0:
            raise ValueError("Fluid density must be positive")

        tmp = (
            (1 - self.rou / rouf)
            * self.mass
            * torch.tensor([gl, 0, 0], device=self.device)
        )
        self.cF = torch.sum(-self.lF * self.lm, dim=0) + tmp

        self.cu = self.cu + self.cF / self.mass * dt
        self.cx = self.cx + self.cu * dt

        self.cT = -torch.sum(
            torch.cross(self.lx - self.cx, self.lF, dim=-1) * self.lm, dim=0
        )
        self.cw = self.cw + self.cT * dt / self.I

    @run_timer
    def update(self, dt):
        self.coord.update(center=self.cx, w=self.cw)
        self.lx = self.coord.cul_point(self._lagrange)

    def from_json(self):
        pass

    def to_str(self, step=0, *args, **kwargs):
        res = f"m={self.mass:.2f}"
        res += "\tcu=" + ",".join([f"{i:.6f}" for i in self.cu])
        res += "\tcx=" + ",".join([f"{i:.6f}" for i in self.cx])
        res += "\tcf=" + ",".join([f"{i:.6f}" for i in self.cF])
        res += "\tlF=" + ",".join(
            [f"{i:.6f}" for i in [self.lF.min(), self.lF.mean(), self.lF.max()]]
        )
        res += "\tcw=" + ",".join([f"{i:.6f}" for i in self.cw])
        res += "\tr=" + self.coord.to_str()
        return res


class Ellipsoid(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra, self.rb, self.rc = (
            self.config.get("a") or 10,
            self.config.get("b") or 10,
            self.config.get("c") or 10,
        )

    def _init(self, dx=1, *args, **kwargs):
        xl, yl, zl = -self.ra - 2 * dx, -self.rb - 2 * dx, -self.rc - 2 * dx
        xr, yr, zr = self.ra + 2 * dx, self.rb + 2 * dx, self.rc + 2 * dx
        self._lagrange = torch.tensor(
            cul_point(
                xl,
                yl,
                zl,
                xr,
                yr,
                zr,
                cul_value=lambda X, Y, Z: X**2 / self.ra**2
                + Y**2 / self.rb**2
                + Z**2 / self.rc**2
                - 1,
                dx=dx,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.mass = torch.tensor(
            4.0 / 3.0 * math.pi * self.ra * self.rb * self.rc,
            device=self.device,
            dtype=torch.float32,
        )
        self.area = torch.tensor(
            4.0 / 3.0 * math.pi * math.pow(self.ra * self.rb * self.rc, 2.0 / 3.0),
            device=self.device,
            dtype=torch.float32,
        )
        self.I = torch.tensor(
            np.array([self.rb * self.rc, self.ra * self.rc, self.ra * self.rb])
            * self.mass.to("cpu").numpy()
            / 5.0,
            device=self.device,
            dtype=torch.float32,
        )


class Sphere(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = self.config.get("r") or 10

    def _init(self, dx=1, *args, **kwargs):
        xl, yl, zl = -self.r - 2 * dx, -self.r - 2 * dx, -self.r - 2 * dx
        xr, yr, zr = self.r + 2 * dx, self.r + 2 * dx, self.r + 2 * dx
        self._lagrange = torch.tensor(
            cul_point(
                xl,
                yl,
                zl,
                xr,
                yr,
                zr,
                cul_value=lambda X, Y, Z: X**2 / self.r**2
                + Y**2 / self.r**2
                + Z**2 / self.r**2
                - 1,
                dx=dx,
                device=self.device,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.mass = torch.tensor(
            4.0 / 3.0 * math.pi * self.r * self.r * self.r,
            device=self.device,
            dtype=torch.float32,
        )
        self.area = torch.tensor(
            4.0 * math.pi * self.r * self.r,
            device=self.device,
            dtype=torch.float32,
        )

        self.I = torch.tensor(
            np.array([self.r * self.r, self.r * self.r, self.r * self.r])
            * self.mass.to("cpu").numpy()
            / 5.0,
            device=self.device,
            dtype=torch.float32,
        )


def example():
    ellipsoid = Ellipsoid(config=ParticleConfig())
    ellipsoid.init()
    ellipsoid.update()

    print(ellipsoid.lx)
    print("#####")
    print(ellipsoid.lx)


# example()
