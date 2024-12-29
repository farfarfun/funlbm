import numpy as np
import torch
from funutil import deep_get
from torch import Tensor

from funlbm.base import Worker
from funlbm.config.base import BoundaryConfig, BaseConfig
from funlbm.parameter import Param


class FlowConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        self.size = np.zeros(3)
        self.param = {}
        self.param_type: str = "D3Q19"
        self.boundary: BoundaryConfig = None

        self.gl = 0.0
        self.Re = 10
        self.mu = 10
        super().__init__(*args, **kwargs)

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.size = np.array(
            deep_get(config_json, "size") or [100, 100, 100], dtype=int
        )
        self.param = deep_get(config_json, "param") or self.param
        self.boundary = BoundaryConfig().from_json(
            deep_get(config_json, "boundary") or {}
        )
        self.param_type = deep_get(config_json, "param_type") or self.param_type

        self.Re = float(deep_get(self.param, "Re") or self.Re)
        self.mu = float(deep_get(self.param, "mu") or self.mu)
        self.gl = float(deep_get(self.param, "gl") or self.gl)


class Flow(Worker):
    def __init__(self, param: Param, config: FlowConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
