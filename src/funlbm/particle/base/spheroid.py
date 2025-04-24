import numpy as np

from funlbm.util import logger

from .ellipsoid import Ellipsoid


class Spheroid(Ellipsoid):
    def __init__(self, ra=None, rb=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra = ra or self.config.get("a") or 20
        self.rb = rb or self.config.get("b") or 10
        self.rc = rb or self.config.get("b") or 10

    def compute_vector(self) -> np.array:
        self.update()
        B1 = self.config.get("B1")

        lx = self._lagrange.cpu().numpy()
        x, y, z = lx[:, 0], lx[:, 1], lx[:, 2]
        if B1 is None or B1 == 0:
            return np.zeros_like(lx)

        # numerator = (x / self.ra) - 1
        # denominator = (x**2 / self.ra**4) + (y**2 / self.rb**4) + (z**2 / self.rc**4)
        # scale = numerator / denominator
        # angle = np.array([self.ra - x, -y, -z]) - scale * np.array(
        #     [x / self.ra**2, y / self.rb**2, z / self.rc**2]
        # )
        angle = np.array([(self.ra**2 - x**2) / -np.abs(x), -y, -z])

        """
        将笛卡尔坐标 (x, y, z) 转换为长球体坐标 (tau, xi, phi).
        """
        c = np.sqrt(abs(self.ra**2 - self.rb**2)) + 0.00000000000001
        r1 = np.sqrt(x**2 + y**2 + (z + c) ** 2)
        r2 = np.sqrt(x**2 + y**2 + (z - c) ** 2)
        tau = (r1 + r2) / (2 * c)
        xi = z / (c * tau)
        phi = np.arctan2(y, x)

        """
        计算椭球 Squirmer 的表面速度 u_s.
        """

        # 极角方向的单位向量 e_xi
        e_xi = np.array(
            [
                -xi * np.sqrt(1 - xi**2) * np.cos(phi),
                -xi * np.sqrt(1 - xi**2) * np.sin(phi),
                np.sqrt(1 - xi**2),
            ]
        )
        tau0 = self.ra / (np.sqrt(self.ra**2 - self.rb**2))

        # 表面速度公式
        size = (
            -B1
            * tau0
            * np.sqrt(1 - xi**2)
            * np.sqrt(tau0**2 - xi**2)
            * (1 + (self.config.get("beta") or 1) * xi)
            * e_xi
        )
        U0 = B1 * tau0 * (tau0 - (tau0**2 - 1) * 0.5 * np.log((tau0 + 1) / (tau0 - 1)))
        logger.warning(f"理想游动速度为:{U0}")
        return np.transpose(angle / np.linalg.norm(angle, axis=0) * size)
