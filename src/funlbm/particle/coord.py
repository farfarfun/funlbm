import numpy as np
import torch
from funutil import deep_get
from scipy.spatial.transform import Rotation as R

from funlbm.base import Worker
from funlbm.config.base import BaseConfig
from funlbm.util import logger


class CoordConfig(BaseConfig):
    def __init__(self, alpha=np.pi / 2, beta=0, gamma=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center = [0, 0, 0]
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.center = deep_get(config_json, "center") or self.center
        self.alpha = deep_get(config_json, "alpha") or self.alpha
        self.beta = deep_get(config_json, "beta") or self.beta
        self.gamma = deep_get(config_json, "gamma") or self.gamma


class Coordinate(Worker):
    """
    坐标系
    center: 中心点
    alpha,beta,gamma,三维初始旋转角度
    w: 旋转角

    """

    def __init__(self, config: CoordConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = config or CoordConfig()
        self.center = torch.tensor(
            config.center, device=self.device, dtype=torch.float32
        )
        self.w = torch.Tensor([config.alpha, config.beta, config.gamma])
        self.rotation = R.from_euler("xyz", self.w)

    def cul_point(self, points):
        return (
            torch.Tensor(self.rotation.apply(points), device=self.device) + self.center
        )

    def update(self, cw, *args, **kwargs):
        """
        https://www.cnblogs.com/QiQi-Robotics/p/14562475.html
        :param dw:增量旋转角度
        :param dt:
        :return:
        """

        if cw is None:
            logger.error("dw cannot be None")
            return
        self.w += cw
        self.rotation = R.from_euler("xyz", self.w)


def example():
    print(
        Coordinate(CoordConfig(alpha=np.pi / 2.0, beta=0, gamma=0)).cul_point([1, 0, 1])
    )


# example()
