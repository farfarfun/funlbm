import torch
from torch import Tensor

from funlbm.config import FlowConfig
from funlbm.parameter import Par3D


class Flow(object):
    def __init__(self, config=None, device="mps", *args, **kwargs):
        self.device = device
        # 坐标
        self.x: Tensor = torch.ones([1])
        # 力密度
        self.f: Tensor = torch.ones([1])
        # 速度
        self.u: Tensor = torch.ones([1])
        # 压强
        self.p: Tensor = torch.ones([1])
        # 密度
        self.rou: Tensor = torch.ones([1])
        # 剪切率相关的变量
        self.gama: Tensor = torch.ones([1])

        self.FOL: Tensor = torch.ones([1])

        self.tau: Tensor = torch.ones([1])

        self.config = config or FlowConfig()
        self.param = Par3D(device=self.device)
