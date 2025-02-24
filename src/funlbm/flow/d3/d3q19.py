from ..base import Param
from .base import FlowD3


class FlowD3Q19(FlowD3):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],  # 静止粒子
            # 面心方向
            [1, 0, 0],
            [-1, 0, 0],  # x方向
            [0, 1, 0],
            [0, -1, 0],  # y方向
            [0, 0, 1],
            [0, 0, -1],  # z方向
            # 棱心方向
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],  # xy平面
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],  # yz平面
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],  # xz平面
        ]

        w = [
            [
                1.0 / 3,  # 静止粒子权重
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,  # 面心权重
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,  # 棱心权重(xy)
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,  # 棱心权重(yz)
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
            ]
        ]  # 棱心权重(xz)

        # 构建速度方向的映射关系
        map = dict([(",".join([str(i) for i in xyz]), i) for i, xyz in enumerate(e)])
        vertex_reverse = [
            map[",".join([str(int(-1 * i)) for i in e[index]])]
            for index in range(len(e))
        ]
        super().__init__(
            param=Param(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs),
            *args,
            **kwargs,
        )
