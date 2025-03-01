import math

import numpy as np
import torch
from scipy.optimize import fsolve

from .base import Particle, ParticleConfig


def find_intersection(point1, point2, cul_value):
    """
    找到线段与椭球表面的交点。
    point1: 线段起点 (在椭球内部)。
    point2: 线段终点 (在椭球外部)。
    """

    def ellipsoid_equation(t):
        x = point1[0] + t * (point2[0] - point1[0])
        y = point1[1] + t * (point2[1] - point1[1])
        z = point1[2] + t * (point2[2] - point1[2])
        return cul_value(x, y, z)

    # 使用数值方法求解交点
    t = fsolve(ellipsoid_equation, 0.5)[0]
    x = point1[0] + t * (point2[0] - point1[0])
    y = point1[1] + t * (point2[1] - point1[1])
    z = point1[2] + t * (point2[2] - point1[2])
    return np.array([x, y, z])


def generate_uniform_points_on_ellipsoid(
    xl, yl, zl, xr, yr, zr, cul_value, dx=0.5, *args, **kwargs
):
    """
    在椭球表面上生成均匀分布的点。

    参数:
        a (float): 椭球的x轴半轴长度。
        b (float): 椭球的y轴半轴长度。
        c (float): 椭球的z轴半轴长度。
        h (float): 空间细分的步长（方块的边长）。

    返回:
        np.ndarray: 椭球表面上的均匀分布点，形状为 (N, 3)。
    """
    # 定义椭球的包围盒范围
    x_range = np.arange(xl, xr, dx)
    y_range = np.arange(yl, yr, dx)
    z_range = np.arange(zl, zr, dx)

    # 存储椭球表面上的点
    surface_points = []

    # 遍历所有方块
    for i in range(len(x_range) - 1):
        for j in range(len(y_range) - 1):
            for k in range(len(z_range) - 1):
                # 当前方块的 8 个顶点
                x0, x1 = x_range[i], x_range[i + 1]
                y0, y1 = y_range[j], y_range[j + 1]
                z0, z1 = z_range[k], z_range[k + 1]
                vertices = [
                    (x0, y0, z0),
                    (x0, y0, z1),
                    (x0, y1, z0),
                    (x0, y1, z1),
                    (x1, y0, z0),
                    (x1, y0, z1),
                    (x1, y1, z0),
                    (x1, y1, z1),
                ]

                # 判断顶点是否在椭球内部或外部
                inside = [cul_value(x, y, z) < 0 for x, y, z in vertices]
                if any(inside) and not all(inside):
                    # 当前方块与椭球表面相交
                    # 找到所有与椭球表面相交的边
                    intersection_points = []
                    for idx1 in range(len(vertices)):
                        for idx2 in range(idx1 + 1, len(vertices)):
                            if inside[idx1] != inside[idx2]:
                                # 找到交点
                                point1 = np.array(vertices[idx1])
                                point2 = np.array(vertices[idx2])
                                intersection = find_intersection(
                                    point1, point2, cul_value
                                )
                                intersection_points.append(intersection)

                    # 计算切面的重心
                    if len(intersection_points) >= 3:
                        centroid = np.mean(intersection_points, axis=0)
                        surface_points.append(centroid)

    return np.array(surface_points)


class Ellipsoid(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra, self.rb, self.rc = (
            self.config.get("a") or 10,
            self.config.get("b") or 10,
            self.config.get("c") or 10,
        )

    def _init(self, dx=1, *args, **kwargs):
        xl, yl, zl = -self.ra - 2 * dx, -self.rb - 2 * dx, -self.rc - 2 * dx
        xr, yr, zr = self.ra + 2 * dx, self.rb + 2 * dx, self.rc + 2 * dx
        self._lagrange = torch.tensor(
            generate_uniform_points_on_ellipsoid(
                xl,
                yl,
                zl,
                xr,
                yr,
                zr,
                cul_value=lambda X, Y, Z: X**2 / self.ra**2
                + Y**2 / self.rb**2
                + Z**2 / self.rc**2
                - 1,
                dx=dx,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.mass = torch.tensor(
            4.0 / 3.0 * math.pi * self.ra * self.rb * self.rc,
            device=self.device,
            dtype=torch.float32,
        )
        self.area = torch.tensor(
            4.0 / 3.0 * math.pi * math.pow(self.ra * self.rb * self.rc, 2.0 / 3.0),
            device=self.device,
            dtype=torch.float32,
        )
        self.I = torch.tensor(
            np.array([self.rb * self.rc, self.ra * self.rc, self.ra * self.rb])
            * self.mass.to("cpu").numpy()
            / 5.0,
            device=self.device,
            dtype=torch.float32,
        )


def example():
    ellipsoid = Ellipsoid(config=ParticleConfig())
    ellipsoid.init()
    ellipsoid.update()

    print(ellipsoid.lx)
    print("#####")
    print(ellipsoid.lx)
