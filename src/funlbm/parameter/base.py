import math

import numpy as np
import torch


class Par:
    def to_device(self, device="mps"):
        self.e = self.e.to(device)
        self.w = self.w.to(device)
        self.cs = self.cs.to(device)

    def __init__(self, device="cpu"):
        self.device = device
        self.e = None
        self.w = None
        self.cs = torch.tensor(np.array(1.0 / math.sqrt(3)), device=self.device, dtype=torch.float32)


class Par3D(Par):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e = np.array(
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

        self.w = np.array(
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
        self.e = torch.tensor(self.e, device=self.device, dtype=torch.float32)
        self.w = torch.tensor(self.w, device=self.device, dtype=torch.float32)

        map = dict([(",".join([str(i) for i in xyz.tolist()]), i) for i, xyz in enumerate(self.e)])
        res = [
            map[",".join([str(round(-1 * i) * 1.0) for i in self.e[index].tolist()])]
            for index in range(self.e.shape[0])
        ]
        self.map = np.array(res)

    def bound_index(self, axis, value):
        return [i for i, e in enumerate(self.e) if e[axis] == value]

    def bound_index_map(self, index):
        return [self.map[i] for i in index]

    def e_map(self, index):
        return self.map[index]
