import numpy as np

from funlbm.util import logger

from .ellipsoid import Ellipsoid


class Spheroid(Ellipsoid):
    """
    https://darkchat.yuque.com/org-wiki-darkchat-gfaase/uvmi28/wlbp0rf6bck1ppfm
    """

    def __init__(self, ra=None, rb=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra = ra or self.config.get("a") or 10
        self.rb = ra or self.config.get("a") or 10
        self.rc = rb or self.config.get("b") or 20

    def compute_vector(self) -> np.array:
        self.update()
        B1 = self.config.get("B1") or 0
        beta = self.config.get("beta") or 0

        lx = self._lagrange.cpu().numpy()
        x, y, z = lx[:, 0], lx[:, 1], lx[:, 2]
        if B1 is None or B1 == 0:
            return np.zeros_like(lx)

        angle = np.array([-x, -y, (self.rc**2 - z**2) / -np.abs(z)])

        """
        将笛卡尔坐标 (x, y, z) 转换为长球体坐标 (tau, xi, phi).
        """

        xi = np.arccos(z / np.sqrt(x**2 + y**2 + z**2 - self.ra**2))

        """
        计算椭球 Squirmer 的表面速度 u_s.
        """
        tau0 = self.rc / (np.sqrt(self.rc**2 - self.ra**2))

        # 表面速度公式
        size = -B1 * tau0 * np.sqrt(1 - xi**2) * np.sqrt(tau0**2 - xi**2) * (1 + beta * xi)
        U0 = B1 * tau0 * (tau0 - (tau0**2 - 1) * 0.5 * np.log((tau0 + 1) / (tau0 - 1)))
        logger.warning(f"理想游动速度为:{U0}")
        return np.transpose(angle / np.linalg.norm(angle, axis=0) * size)
