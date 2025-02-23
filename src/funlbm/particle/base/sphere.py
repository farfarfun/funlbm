import math

import numpy as np
import torch

from .base import Particle
from .ellipsoid import generate_uniform_points_on_ellipsoid


class Sphere(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = self.config.get("r") or 10

    def _init(self, dx=1, *args, **kwargs):
        xl, yl, zl = -self.r - 2 * dx, -self.r - 2 * dx, -self.r - 2 * dx
        xr, yr, zr = self.r + 2 * dx, self.r + 2 * dx, self.r + 2 * dx
        self._lagrange = torch.tensor(
            generate_uniform_points_on_ellipsoid(
                xl,
                yl,
                zl,
                xr,
                yr,
                zr,
                cul_value=lambda X, Y, Z: X**2 / self.r**2
                + Y**2 / self.r**2
                + Z**2 / self.r**2
                - 1,
                dx=dx,
                device=self.device,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.mass = torch.tensor(
            4.0 / 3.0 * math.pi * self.r * self.r * self.r,
            device=self.device,
            dtype=torch.float32,
        )
        self.area = torch.tensor(
            4.0 * math.pi * self.r * self.r,
            device=self.device,
            dtype=torch.float32,
        )

        self.I = torch.tensor(
            np.array([self.r * self.r, self.r * self.r, self.r * self.r])
            * self.mass.to("cpu").numpy()
            / 5.0,
            device=self.device,
            dtype=torch.float32,
        )
