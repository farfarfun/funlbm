import math

import torch


class Param:
    def __init__(self, e, w, vertex_reverse, device="cpu"):
        self.device = device
        self.vertex_reverse = vertex_reverse
        self.e = torch.tensor(e, device=self.device, dtype=torch.float32)
        self.w = torch.tensor(w, device=self.device, dtype=torch.float32)
        self.cs = torch.tensor(math.sqrt(1.0 / 3), device=self.device, dtype=torch.float32)

    def vertex_index(self, axis, value):
        return [i for i, e in enumerate(self.e) if e[axis] == value]

    def index_reverse(self, index):
        if isinstance(index, int):
            return self.vertex_reverse[index]
        return [self.vertex_reverse[i] for i in index]


class ParamD3Q27(Param):
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
