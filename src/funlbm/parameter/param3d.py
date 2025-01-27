from funlbm.util import logger

from .base import Param


class ParamD3Q27(Param):
    """D3Q27模型参数类

    实现了27个离散速度方向的格子玻尔兹曼模型参数
    包括:
    - 离散速度向量
    - 权重系数
    - 速度方向映射关系
    """

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
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


class ParamD3Q19(Param):
    """D3Q19模型参数类

    实现了19个离散速度方向的格子玻尔兹曼模型参数
    包括:
    - 离散速度向量
    - 权重系数
    - 速度方向映射关系
    """

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
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


class ParamD3Q15(Param):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ]

        w = [
            [
                2.0 / 9,
                1.0 / 9,
                1.0 / 9,
                1.0 / 9,
                1.0 / 9,
                1.0 / 9,
                1.0 / 9,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
                1.0 / 72,
            ]
        ]
        map = dict([(",".join([str(i) for i in xyz]), i) for i, xyz in enumerate(e)])
        vertex_reverse = [
            map[",".join([str(int(-1 * i)) for i in e[index]])]
            for index in range(len(e))
        ]
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


class ParamD3Q13(Param):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, 1],
            [1, 0, -1],
            [0, 1, 1],
            [0, 1, -1],
            [-1, -1, 0],
            [-1, 1, 0],
            [-1, 0, -1],
            [-1, 0, 1],
            [0, -1, -1],
            [0, -1, 1],
        ]

        w = [
            [
                1.0 / 2,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
                1.0 / 24,
            ]
        ]
        map = dict([(",".join([str(i) for i in xyz]), i) for i, xyz in enumerate(e)])
        vertex_reverse = [
            map[",".join([str(int(-1 * i)) for i in e[index]])]
            for index in range(len(e))
        ]
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


def parse_3d_param(param_type, *args, **kwargs):
    """解析3D模型参数

    根据参数类型创建对应的参数对象

    Args:
        param_type: 参数类型,支持"D3Q27"/"D3Q19"/"D3Q15"/"D3Q13"

    Returns:
        Param: 参数对象

    Raises:
        ValueError: 当参数类型不支持时
    """
    logger.info(f"param_type={param_type}")
    if param_type == "D3Q27":
        return ParamD3Q27(*args, **kwargs)
    elif param_type == "D3Q19":
        return ParamD3Q19(*args, **kwargs)
    elif param_type == "D3Q15":
        return ParamD3Q15(*args, **kwargs)
    elif param_type == "D3Q13":
        return ParamD3Q13(*args, **kwargs)
    else:
        raise ValueError("Unknown parameter type: {}".format(param_type))
