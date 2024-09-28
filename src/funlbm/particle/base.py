import logging

import numpy as np
import transforms3d as tfs

from funlbm.config import ParticleConfig, CoordConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("funlbm")
logger.setLevel(logging.INFO)


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
    def __init__(self, config: CoordConfig = None, *args, **kwargs):
        config = config or CoordConfig()
        self.center = config.center
        self.r = tfs.euler.euler2mat(
            config.alpha, config.beta, config.gamma
        )  # alpha,beta,gamma是旋转角度
        self.w = np.zeros([3])

    def cul_point(self, points):
        return np.matmul(points, self.r) + self.center

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
        tmp = np.array(
            [[1, -wz * dt, wy * dt], [wz * dt, 1, -wx * dt, -wy * dt, wx * dt, 1]]
        )
        self.r = np.matmul(tmp, self.r)


class Particle:
    def __init__(self, config: ParticleConfig = None, *args, **kwargs):
        self.config: ParticleConfig = config or ParticleConfig()
        self.coord: Coordinate = Coordinate(config=self.config.coord_config)

        # 颗粒质量[1]
        self.mass = None
        # 惯性矩
        self.I = None
        # 颗粒密度[1]
        self.rou = None
        # 颗粒方向[i,j,k]
        self.angle = None

        # 质心坐标[i,j,k]
        self.cx = self.config.coord_config.center
        # 质心半径[a,b,b]
        self.cr = 5 * np.ones(5)
        # 质心速度[i,j,k]
        self.cu = np.zeros(3) + 0.01
        # 质心角速度[i,j,k]
        self.cw = np.zeros(3)
        # 质心合外力
        self.cF = np.zeros(3)
        # 质心合外力
        self.cT = np.zeros(3)

        self._lagrange: np.ndarray = np.zeros([0])
        # 拉格朗日点的坐标[m,i,3]
        self.lx: np.ndarray = np.zeros([0])
        # 拉格朗日点上的力[m,i,3]
        self.lF = np.ndarray = np.zeros([0])
        # 拉格朗日点上的力矩[m,i,3]
        self.lT = np.ndarray = np.zeros([0])
        # 拉格朗日点的质量
        self.lm = np.ndarray = np.zeros([0])
        # 拉格朗日点速度[m,i,3]
        self.lu = np.ndarray = np.zeros([0])
        # 拉格朗日点速度[m,i,3]
        self.lrou = np.ndarray = np.zeros([0])

    def _init(self, dx=0.2, *args, **kwargs):
        raise NotImplemented("还没实现")

    def init(self, *args, **kwargs):
        self.rou = float(self.config.get("rou", 1.0))
        self._init(*args, **kwargs)
        shape = self._lagrange.shape
        self.lx = np.zeros(shape)
        self.lF = np.zeros(shape)
        self.lT = np.zeros(shape)
        self.lu = np.zeros(shape)
        self.lm = np.ones((shape[0], 1)) * self.mass / shape[0]
        self.lrou = np.zeros((shape[0], 1))

    def update_from_lar(self, dt=0.1):
        self.cF = np.sum(-self.lF * self.lm, axis=0)

        self.cT = np.sum(-self.lT * self.lm, axis=0)
        self.cu = self.cu + self.cF / self.mass
        self.cx = self.cx + self.cu * dt
        self.cw = self.cw + self.cT * dt / self.I
        # self.cw = self.cw * 0  # TODO 暂时去掉角速度，方便排查

    def update(self, dw=None, dt=None):
        self.coord.update(dw, dt)
        self.lx = self.coord.cul_point(self._lagrange)
        self.lu = self.cu + np.cross(self.cw, self.lx - self.cx)

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
        self._lagrange = cul_point(
            xl,
            yl,
            zl,
            xr,
            yr,
            zr,
            cul_value=lambda X, Y, Z: X**2 / 5 + Y**2 / 5 + Z**2 / 5 - 1,
            dx=dx,
        )
        self.mass = 4.0 / 3 * self.ra * self.rb * self.rc
        self.I = (
            np.array([self.rb * self.rc, self.ra * self.rc, self.ra * self.rb])
            * self.mass
            / 5.0
        )


def example():
    ellipsoid = Ellipsoid(config=ParticleConfig())
    ellipsoid.init()
    ellipsoid.update()

    print(ellipsoid.lx)
    print("#####")
    print(ellipsoid.lx)


# example()
