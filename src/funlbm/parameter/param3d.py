from .base import Param


class ParamD3Q19(Param):
    def __init__(self, *args, **kwargs):
        e = [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],  # 1
            [0, 1, 0],
            [0, -1, 0],  # 3
            [0, 0, 1],
            [0, 0, -1],  # 5
            [1, 1, 0],
            [-1, -1, 0],  # 7
            [1, -1, 0],
            [-1, 1, 0],  # 9
            [1, 0, 1],
            [-1, 0, -1],  # 11
            [1, 0, -1],
            [-1, 0, 1],  # 13
            [0, 1, 1],
            [0, -1, -1],  # 15
            [0, 1, -1],
            [0, -1, 1],
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
