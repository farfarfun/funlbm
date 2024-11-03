import math

import torch
from funutil.cache import cache


class Param:
    def __init__(self, e, w, vertex_reverse, device="cpu"):
        self.device = device
        self.vertex_reverse = vertex_reverse
        self.e = torch.tensor(e, device=self.device, dtype=torch.float32)
        self.w = torch.tensor(w, device=self.device, dtype=torch.float32)
        self.cs = torch.tensor(math.sqrt(1.0 / 3), device=self.device, dtype=torch.float32)

    @cache
    def eT(self) -> torch.Tensor:
        return self.e.t()

    @cache
    def vertex_index(self, axis, value):
        return [i for i, e in enumerate(self.e) if e[axis] == value]

    def index_reverse(self, index):
        if isinstance(index, int):
            return self.vertex_reverse[index]
        return [self.vertex_reverse[i] for i in index]

    @property
    def e_dim(self):
        return self.e.shape[0]
