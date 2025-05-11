import math

import numpy as np
import torch

from .ellipsoid import Ellipsoid, generate_uniform_points_on_ellipsoid


class Sphere(Ellipsoid):
    def __init__(self, r=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r or self.config.get("r") or 10
        self.ra = self.r
        self.rb = self.r
        self.rc = self.r

    def _init(self, dx=1, *args, **kwargs):
        super()._init(dx=dx, *args, **kwargs)
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
