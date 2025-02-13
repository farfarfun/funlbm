import numpy as np
import torch
from funutil import deep_get
from torch import Tensor

from funlbm.base import Worker
from funlbm.config.base import BaseConfig, BoundaryConfig
from funlbm.parameter import Param
from funlbm.util import tensor_format


class FlowConfig(BaseConfig):
    """流体配置类

    Attributes:
        size: 计算域大小,形状为(3,)的数组
        param: 参数字典
        param_type: 参数类型,默认为"D3Q19"
        boundary: 边界配置
        gl: 重力加速度
        Re: 雷诺数
        mu: 动力粘度
    """

    def __init__(self, *args, **kwargs):
        self.size: np.ndarray = np.zeros(3)  # 计算域大小
        self.param: dict = {}  # 参数字典
        self.param_type: str = "D3Q19"  # 参数类型
        self.boundary: BoundaryConfig = None  # 边界配置

        self.gl: float = 0.0  # 重力加速度
        self.Re: float = 10  # 雷诺数
        self.mu: float = 10  # 动力粘度
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
    """流场基类

    实现了流场计算的基本功能

    属性:
        param (Param): 参数对象
        config (FlowConfig): 配置对象
        x (Tensor): 坐标张量
        f (Tensor): 力密度分布函数
        feq (Tensor): 平衡态分布函数
        u (Tensor): 速度场
        p (Tensor): 压力场
        rou (Tensor): 密度场
        gama (Tensor): 剪切率
        FOL (Tensor): 力项
        tau (Tensor): 松弛时间
    """

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

    def init(self, *args, **kwargs) -> None:
        """初始化流场"""
        raise NotImplementedError("not implemented")

    def update_u_rou(self, *args, **kwargs) -> None:
        """更新速度和密度场"""
        raise NotImplementedError("not implemented")

    def cul_equ(self, tau: Tensor = None, *args, **kwargs) -> None:
        """计算平衡态分布函数"""
        raise NotImplementedError("not implemented")

    def cul_equ2(self, *args, **kwargs) -> None:
        """计算第二种平衡态分布函数"""
        raise NotImplementedError("not implemented")

    def f_stream(self, *args, **kwargs) -> None:
        """执行流动计算"""
        raise NotImplementedError("not implemented")

    def __repr__(self) -> str:
        """返回流场对象的字符串表示"""
        return f"{self.__class__.__name__}(size={tuple(self.config.size)}, Re={self.config.Re}, mu={self.config.mu})"

    def to_json(self):
        return {
            "f": tensor_format(
                [
                    self.f.min(),
                    self.f.mean(),
                    self.f.max(),
                ]
            ),
            "u": tensor_format(
                [
                    self.u.min(),
                    self.u.mean(),
                    self.u.max(),
                ]
            ),
            "rho": tensor_format(
                [
                    self.rou.min(),
                    self.rou.mean(),
                    self.rou.max(),
                ]
            ),
        }
