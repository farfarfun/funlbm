import numpy as np

from funlbm.config import FlowConfig
from funlbm.parameter import Par3D


class Flow(object):
    def __init__(self, config=None, *args, **kwargs):
        # 坐标
        self.x: np.ndarray = None
        # 力密度
        self.f: np.ndarray = None
        # 速度
        self.u: np.ndarray = None
        # 压强
        self.p: np.ndarray = None
        # 密度
        self.rou: np.ndarray = None
        # 剪切率相关的变量
        self.gama: np.ndarray = None

        self.FOL: np.ndarray = None

        self.tau: np.ndarray = None

        self.param = Par3D()
        self.config = config or FlowConfig()
