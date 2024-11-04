import torch
from torch import Tensor

from funlbm.config import FlowConfig
from funlbm.parameter import Param


class Flow(object):
    def __init__(self, param: Param, config: FlowConfig, device="cpu", *args, **kwargs):
        self.device = device
        self.param: Param = param
        self.config: FlowConfig = config

        # 坐标
        self.x: Tensor = torch.zeros([1])
        # 力密度
        self.f: Tensor = torch.zeros([1])
        self.feq: Tensor = torch.zeros([1])
        # 速度
        self.u: Tensor = torch.zeros([1])
        # 压强
        self.p: Tensor = torch.zeros([1])
        # 密度
        self.rou: Tensor = torch.zeros([1])
        # 剪切率相关的变量
        self.gama: Tensor = torch.zeros([1])

        self.FOL: Tensor = torch.zeros([1])

        self.tau: Tensor = torch.zeros([1])

    def init(self, *args, **kwargs):
        raise NotImplementedError("not implemented")

    def update_u_rou(self, *args, **kwargs):
        raise NotImplementedError("not implemented")

    def cul_equ(self, tau=None, *args, **kwargs):
        raise NotImplementedError("not implemented")

    def cul_equ2(self, *args, **kwargs):
        raise NotImplementedError("not implemented")

    def f_stream(self, *args, **kwargs):
        raise NotImplementedError("not implemented")
