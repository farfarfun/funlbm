import funutil
import numpy as np
import torch
import transforms3d as tfs
from funutil import run_timer
from torch import Tensor

from funlbm.config import CoordConfig, ParticleConfig

logger = funutil.getLogger("funlbm")


def cul_point(xl, yl, zl, xr, yr, zr, cul_value, dx=0.1):
    x = np.linspace(xl, xr, int((xr - xl) / dx))
    y = np.linspace(yl, yr, int((yr - yl) / dx))
    z = np.linspace(zl, zr, int((zr - zl) / dx))

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    value = cul_value(X, Y, Z)
    inner = (value < 0) * 1
    outer = (value > 0) * 2
    S = inner + outer

    T = S[:-1, :-1, :-1]
    T = T + S[1:, :-1, :-1] + S[:-1, 1:, :-1] + S[:-1, :-1, 1:]
    T = T + S[:-1, 1:, 1:] + S[1:, :-1, 1:] + S[1:, 1:, :-1]
    T = T + S[1:, 1:, 1:]

    xm = T.mean(axis=2).mean(axis=1)
    ym = T.mean(axis=2).mean(axis=0)
    zm = T.mean(axis=0).mean(axis=0)
    result = []
    for i in range(T.shape[0]):
        if xm[i] == 16:
            continue
        for j in range(T.shape[1]):
            if ym[j] == 16:
                continue
            for k in range(T.shape[2]):
                if zm[j] == 16:
                    continue
                if T[i, j, k] == 16:
                    continue

                # print(k, T[i, j, k], X[i][j][k], Y[i][j][k], Z[i][j][k])
                result.append([X[i][j][k], Y[i][j][k], Z[i][j][k]])
    logger.info(f"lagrange size:{len(result)}")
    return np.array(result, dtype=np.float32)


class Coordinate:
    def __init__(self, config: CoordConfig = None, device="cpu", *args, **kwargs):
        self.device = device
        config = config or CoordConfig()
        self.center = torch.tensor(config.center, device=self.device, dtype=torch.float32)
        # alpha,beta,gamma是旋转角度
        self.r = torch.tensor(
            tfs.euler.euler2mat(config.alpha, config.beta, config.gamma), device=self.device, dtype=torch.float32
        )
        self.w = torch.zeros([3])

    def cul_point(self, points):
        return torch.matmul(points, self.r) + self.center

    def update(self, dw=None, dt=None):
        """
        https://www.cnblogs.com/QiQi-Robotics/p/14562475.html
        :param dw:
        :param dt:
        :return:
        """
        if dw is None or dt is None:
            return
        self.w += dw
        wx, wy, wz = self.w
        tmp = torch.tensor(
            np.array([[1, -wz * dt, wy * dt], [wz * dt, 1, -wx * dt, -wy * dt, wx * dt, 1]]),
            device=self.device,
            dtype=torch.float32,
        )
        self.r = torch.matmul(tmp, self.r)


class Particle:
    def __init__(self, config: ParticleConfig = None, device="mps", *args, **kwargs):
        self.device = device
        self.config: ParticleConfig = config or ParticleConfig()
        self.coord: Coordinate = Coordinate(config=self.config.coord_config, device=self.device, *args, **kwargs)

        # 颗粒质量[1]
        self.mass = None
        # 惯性矩
        self.I = None
        # 颗粒密度[1]
        self.rou = None
        # 颗粒方向[i,j,k]
        self.angle = None

        # 质心坐标[i,j,k]
        self.cx = torch.tensor(self.config.coord_config.center, device=self.device, dtype=torch.float32)
        # 质心半径[a,b,b]
        self.cr = 5 * torch.ones(5, device=self.device, dtype=torch.float32)
        # 质心速度[i,j,k]
        self.cu = torch.zeros(3, device=self.device, dtype=torch.float32) + 0.01
        # 质心角速度[i,j,k]
        self.cw = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cF = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cT = torch.zeros(3, device=self.device, dtype=torch.float32)

        self._lagrange: Tensor = torch.zeros([0])
        # 拉格朗日点的坐标[m,i,3]
        self.lx: Tensor = torch.zeros([0])
        # 拉格朗日点上的力[m,i,3]
        self.lF: Tensor = torch.zeros([0])
        # 拉格朗日点上的力矩[m,i,3]
        self.lT: Tensor = torch.zeros([0])
        # 拉格朗日点的质量
        self.lm: Tensor = torch.zeros([0])
        # 拉格朗日点速度[m,i,3]
        self.lu: Tensor = torch.zeros([0])
        # 拉格朗日点速度[m,i,3]
        self.lrou: Tensor = torch.zeros([0])

    def _init(self, dx=0.2, *args, **kwargs):
        raise NotImplementedError("还没实现")

    def init(self, *args, **kwargs):
        self.rou = float(self.config.get("rou", 1.0))
        self._init(*args, **kwargs)
        shape = self._lagrange.shape
        self.lx = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.lF = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.lT = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.lu = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.lm = torch.ones((shape[0], 1), device=self.device, dtype=torch.float32) * self.mass / shape[0]
        self.lrou = torch.zeros((shape[0], 1), device=self.device, dtype=torch.float32)

    @run_timer
    def update_from_lar(self, dt=0.1):
        self.cF = torch.sum(-self.lF * self.lm, dim=0)
        self.cT = torch.sum(-self.lT * self.lm, dim=0)
        self.cu = self.cu + self.cF / self.mass
        self.cx = self.cx + self.cu * dt
        self.cw = self.cw + self.cT * dt / self.I
        # self.cw = self.cw * 0  # TODO 暂时去掉角速度，方便排查

    @run_timer
    def update(self, dw=None, dt=None):
        self.coord.update(dw, dt)
        self.lx = self.coord.cul_point(self._lagrange)
        self.lu = self.cu + torch.cross(self.cw.unsqueeze(0), self.lx - self.cx, dim=-1)

    def from_json(self):
        pass

    def to_str(self):
        res = f"m={self.mass:2f}"
        res += "\tu=" + ",".join([f"{i:6f}" for i in self.cu])
        res += "\tx=" + ",".join([f"{i:6f}" for i in self.cx])
        res += "\tf=" + ",".join([f"{i:6f}" for i in self.cF])
        res += "\tw=" + ",".join([f"{i:6f}" for i in self.cw])
        return res


class Ellipsoid(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra, self.rb, self.rc = (
            self.config.get("a") or 10,
            self.config.get("b") or 10,
            self.config.get("c") or 10,
        )

    def _init(self, dx=0.2, *args, **kwargs):
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
                cul_value=lambda X, Y, Z: X**2 / 5 + Y**2 / 5 + Z**2 / 5 - 1,
                dx=dx,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.mass = torch.tensor(4.0 / 3 * self.ra * self.rb * self.rc, device=self.device, dtype=torch.float32)
        self.I = torch.tensor(
            np.array([self.rb * self.rc, self.ra * self.rc, self.ra * self.rb]) * self.mass.to("cpu").numpy() / 5.0,
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
