import math

import h5py
import numpy as np
import torch
from funutil.cache import cache
from torch import Tensor

from funlbm.base import Worker
from funlbm.config.base import BaseConfig, BoundaryConfig
from funlbm.util import tensor_format


class Param(Worker):
    """格子玻尔兹曼模型参数基类

    提供离散速度、权重系数等基本参数的管理功能

    Args:
        e: 离散速度向量集合
        w: 权重系数集合
        vertex_reverse: 速度方向映射关系

    属性:
        e (Tensor): 离散速度向量
        w (Tensor): 权重系数
        cs (Tensor): 声速
        vertex_reverse (list): 速度方向映射表
    """

    def __init__(self, e, w, vertex_reverse, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_reverse = vertex_reverse
        self.e = torch.tensor(e, device=self.device, dtype=torch.float32)
        self.w = torch.tensor(w, device=self.device, dtype=torch.float32)
        self.cs = torch.tensor(
            math.sqrt(1.0 / 3), device=self.device, dtype=torch.float32
        )

    @cache
    def eT(self) -> torch.Tensor:
        """返回离散速度向量的转置矩阵"""
        return self.e.t()

    @cache
    def vertex_index(self, axis, value):
        """获取指定方向和值的速度索引

        Args:
            axis: 坐标轴(0,1,2)
            value: 速度值

        Returns:
            list: 满足条件的速度索引列表
        """
        return [i for i, e in enumerate(self.e) if e[axis] == value]

    def index_reverse(self, index):
        """获取速度方向的反向索引

        Args:
            index: 速度索引,可以是单个索引或索引列表

        Returns:
            反向速度的索引
        """
        if isinstance(index, int):
            return self.vertex_reverse[index]
        return [self.vertex_reverse[i] for i in index]

    @property
    def e_dim(self):
        """离散速度的维度"""
        return self.e.shape[0]


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

    def __init__(
        self,
        size=None,
        boundary=None,
        param_type: str = "D3Q19",
        gl: float = 0.0,
        Re: float = 10,
        mu: float = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.size: np.ndarray = np.array(size or [100, 100, 100], dtype=int)
        self.param_type: str = param_type
        self.boundary: BoundaryConfig = BoundaryConfig(**boundary or {})

        self.gl: float = gl  # 重力加速度
        self.Re: float = Re  # 雷诺数
        self.mu: float = mu  # 动力粘度


class FlowBase(Worker):
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

        # 坐标,全局不变
        self.x: Tensor = torch.zeros([1])
        # 力密度，每步都变
        self.f: Tensor = torch.zeros([1])
        self.feq: Tensor = torch.zeros([1])
        # 速度，每步都变，可以由f算出来
        self.u: Tensor = torch.zeros([1])
        # 压强，暂无
        self.p: Tensor = torch.zeros([1])
        # 密度，每步都变，可以有f算出
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

    def track(self):
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

    def dump_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or ["x", "f", "p", "u", "rou", "gama", "FOL", "tau", "feq"]
        if "x" in vals:
            self.dump_dataset(group, "x", self.x)
        if "f" in vals:
            self.dump_dataset(group, "f", self.f)
        if "p" in vals:
            self.dump_dataset(group, "p", self.p)
        if "u" in vals:
            self.dump_dataset(group, "u", self.u)
        if "rou" in vals:
            self.dump_dataset(group, "rou", self.rou)
        if "gama" in vals:
            self.dump_dataset(group, "gama", self.gama)
        if "FOL" in vals:
            self.dump_dataset(group, "FOL", self.FOL)
        if "tau" in vals:
            self.dump_dataset(group, "tau", self.tau)
        if "feq" in vals:
            self.dump_dataset(group, "feq", self.feq)

    def load_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or group.keys()
        if "x" in vals and "x" in group.keys():
            self.x = self.load_dataset(group, "x")
        if "f" in vals and "f" in group.keys():
            self.f = self.load_dataset(group, "f")
        if "u" in vals and "u" in group.keys():
            self.u = self.load_dataset(group, "u")
        if "rou" in vals and "rou" in group.keys():
            self.rou = self.load_dataset(group, "rou")
        if "gama" in vals and "gama" in group.keys():
            self.gama = self.load_dataset(group, "gama")
        if "FOL" in vals and "FOL" in group.keys():
            self.FOL = self.load_dataset(group, "FOL")
        if "tau" in vals and "tau" in group.keys():
            self.tau = self.load_dataset(group, "tau")
        if "feq" in vals and "feq" in group.keys():
            self.feq = self.load_dataset(group, "feq")
