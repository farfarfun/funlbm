from ..base import Param
from .base import FlowD3


class FlowD3Q27(FlowD3):
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
            # 顶点方向
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]

        w = [
            [
                8.0 / 27,  # 静止粒子权重
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,  # 面心权重
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,  # 棱心权重(xy)
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,  # 棱心权重(yz)
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,  # 棱心权重(xz)
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,  # 顶点权重
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
            ]
        ]  # 顶点权重

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
