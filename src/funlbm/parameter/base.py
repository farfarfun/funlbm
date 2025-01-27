import math

import torch
from funutil.cache import cache

from funlbm.base import Worker


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
