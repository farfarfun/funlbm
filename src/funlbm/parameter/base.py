import math

import numpy as np


class Par:
    e = None
    w = None
    cs = 1.0 / math.sqrt(3)


class Par3D(Par):
    e = np.array(
        [
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
    )
    w = np.array(
        [
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
    )

    def __init__(self):
        map = dict(
            [
                (",".join([str(i) for i in xyz.tolist()]), i)
                for i, xyz in enumerate(self.e)
            ]
        )
        res = [
            map[",".join([str(-1 * i) for i in self.e[index].tolist()])]
            for index in range(self.e.shape[0])
        ]
        self.map = np.array(res)

    @staticmethod
    def bound_index(axis, value):
        return [i for i, e in enumerate(Par3D.e) if e[axis] == value]

    def bound_index_map(self, index):
        return [self.map[i] for i in index]

    def e_map(self, index):
        return self.map[index]
