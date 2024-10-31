import funutil

from .base import Param

logger = funutil.getLogger("funlbm")


class ParamD3Q27(Param):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
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
                8.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                2.0 / 27,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 54,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
                1.0 / 216,
            ]
        ]
        map = dict([(",".join([str(i) for i in xyz]), i) for i, xyz in enumerate(e)])
        vertex_reverse = [map[",".join([str(int(-1 * i)) for i in e[index]])] for index in range(len(e))]
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


class ParamD3Q19(Param):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
        ]
        w = [
            [
                1.0 / 3,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 18,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
                1.0 / 36,
            ]
        ]
        map = dict([(",".join([str(i) for i in xyz]), i) for i, xyz in enumerate(e)])
        vertex_reverse = [map[",".join([str(int(-1 * i)) for i in e[index]])] for index in range(len(e))]
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
        vertex_reverse = [map[",".join([str(int(-1 * i)) for i in e[index]])] for index in range(len(e))]
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
        vertex_reverse = [map[",".join([str(int(-1 * i)) for i in e[index]])] for index in range(len(e))]
        super().__init__(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs)


def parse_3d_param(param_type, *args, **kwargs):
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
