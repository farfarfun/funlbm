from ..base import Param
from .base import FlowD3


class FlowD3Q13(FlowD3):
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
        super().__init__(
            param=Param(e=e, w=w, vertex_reverse=vertex_reverse, *args, **kwargs),
            *args,
            **kwargs,
        )
