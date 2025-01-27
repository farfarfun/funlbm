import numpy as np
import torch
from funutil import run_timer
from funvtk.hl import gridToVTK, pointsToVTK

from funlbm.flow import FlowD3
from funlbm.particle import Sphere
from funlbm.util import logger

from .base import Config, LBMBase


class LBMD3(LBMBase):
    """3D格子玻尔兹曼方法实现类

    实现了3D流场的基本功能,包括初始化、边界处理等

    Args:
        config: LBM配置对象
    """

    def __init__(self, config: Config, *args, **kwargs):
        flow = FlowD3(config=config.flow_config, *args, **kwargs)
        particles = [Sphere(config=con) for con in config.particles]
        super(LBMD3, self).__init__(
            flow=flow, config=config, particles=particles, *args, **kwargs
        )

    def init(self):
        """初始化流场和颗粒"""
        # 初始化流程
        # 初始化边界条件
        self.flow.init()
        # 初始化颗粒
        self.flow.cul_equ()

        for particle in self.particles:
            particle.init()

    def _calculate_region_bounds(self, particle, n, h):
        """Calculate region bounds for particle interaction."""
        # 保持在GPU上计算
        rl = torch.floor(particle.lx) - (n - 1) * h
        # 限制下界
        rl = torch.clamp(rl, min=0)
        rr = rl + (2 * n - 1) * h + 1
        # 限制上界
        domain_size = torch.tensor(
            self.config.flow_config.size, device=self.device, dtype=torch.float32
        )
        rr = torch.clamp(rr, max=domain_size)
        return rl.to(dtype=torch.int32), rr.to(dtype=torch.int32)

    def _calculate_weight_function(self, flow_x, lar, h):
        """Calculate weight function for particle-fluid interaction."""
        tmp = flow_x - lar
        # 限制权重函数的作用范围
        mask = torch.all(torch.abs(tmp) <= 2 * h, dim=-1, keepdim=True)
        tmp = torch.where(
            mask, (1 + torch.cos(torch.abs(tmp * np.pi / 2 / h))) / 4 / h, 0.0
        )
        w = torch.prod(tmp, dim=-1, keepdim=True).to(dtype=torch.float32)
        # 归一化权重
        w_sum = torch.sum(w)
        if w_sum > 0:
            w = w / w_sum
        return w

    @run_timer
    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        for particle in self.particles:
            rl, rr = self._calculate_region_bounds(particle, n, h=h)
            # TODO 上限没加

            lu = torch.zeros(particle.lu.shape, device=self.device)
            lrou = torch.zeros((particle.lu.shape[0], 1), device=self.device)

            for index, lar in enumerate(particle.lx):
                il, ir = rl[index], rr[index]
                region_x = self.flow.x[il[0] : ir[0], il[1] : ir[1], il[2] : ir[2], :]
                tmp = self._calculate_weight_function(region_x, lar, h)

                lu[index, :] = torch.sum(
                    self.flow.u[il[0] : ir[0], il[1] : ir[1], il[2] : ir[2], :] * tmp
                )
                lrou[index, :] = torch.sum(
                    self.flow.rou[il[0] : ir[0], il[1] : ir[1], il[2] : ir[2], :] * tmp
                )
            u_theta = 0

            particle.lu[:, :] = (
                particle.cu
                + torch.cross(
                    particle.cw.unsqueeze(0), particle.lx - particle.cx, dim=-1
                )
                + u_theta
            )

            particle.lF = 2 * lrou * (particle.lu - lu) / self.config.dt

    @run_timer
    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        for particle in self.particles:
            rl, rr = self._calculate_region_bounds(particle, n, h)

            for index, lar in enumerate(particle.lx):
                il, ir = rl[index], rr[index]
                region_x = self.flow.x[il[0] : ir[0], il[1] : ir[1], il[2] : ir[2], :]
                tmp = self._calculate_weight_function(region_x, lar, h)

                force = tmp * particle.lF[index, :] * particle.lm[index]
                self.flow.FOL[il[0] : ir[0], il[1] : ir[1], il[2] : ir[2], :] = force

    @run_timer
    def particle_to_wall(self, *args, **kwargs):
        """
        Handle particle-wall collisions using a spring force model.
        """
        k0 = 300  # Spring constant
        wall_normals = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            device=self.device,
            dtype=torch.float32,
        )

        domain_bounds = torch.concatenate(
            [
                torch.zeros(self.config.flow_config.size.shape, device=self.device),
                torch.tensor(
                    self.config.flow_config.size,
                    device=self.device,
                    dtype=torch.float32,
                ),
            ]
        )

        for particle in self.particles:
            # Find extreme points of particle
            extreme_indices = torch.concatenate(
                [torch.argmin(particle.lx, dim=0), torch.argmax(particle.lx, dim=0)]
            )

            # Calculate wall distances
            distances = k0 * (
                1
                - abs(
                    particle.lx[extreme_indices][[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]]
                    - domain_bounds
                )
                / (2 * self.config.dx)
            )
            distances = torch.clamp(distances, min=0)

            if distances.max() == 0:
                continue

            logger.info(f"Distance of particle to wall = {distances}")
            particle.lF[extreme_indices] += torch.multiply(
                wall_normals, distances.unsqueeze(1)
            )

    def save(self, step=10, *args, **kwargs):
        if step % self.config.file_config.per_steps > 0:
            return

        shape = self.flow.u.shape
        xf, yf, zf = np.meshgrid(
            range(shape[0]),
            range(shape[1]),
            range(shape[2]),
            indexing="ij",
            sparse=False,
        )
        flow_u = self.flow.u.to("cpu").numpy()
        point_data = {
            "u": (
                flow_u[:, :, :, 0],
                flow_u[:, :, :, 1],
                flow_u[:, :, :, 2],
            ),
        }

        cell_data = {
            "rho": self.flow.rou.to("cpu").numpy()[:, :, :, 0],
            "p": self.flow.p.to("cpu").numpy()[:, :, :, 0],
        }

        # t1 = self.flow.rou[10:-10, 10:-10, 10:-10, :]
        # t1 = cell_data['rho'][5:-5, 5:-5, 5:-5]
        # print("rho", step, t1.max(), t1.min(), t1.max() - t1.min())

        gridToVTK(
            f"{self.config.file_config.vtk_path}/flow_" + str(step).zfill(10),
            xf,
            yf,
            zf,
            pointData=point_data,
            cellData=cell_data,
        )

        for i, particle in enumerate(self.particles):
            lx = particle.lx.to("cpu").numpy()
            xf, yf, zf = lx[:, 0], lx[:, 1], lx[:, 2]
            data = {
                "u": particle.lu.to("cpu").numpy()[:, 0],
            }
            fname = f"{self.config.file_config.vtk_path}/particle_{str(i).zfill(3)}_{str(step).zfill(10)}"
            pointsToVTK(fname, xf, yf, zf, data=data)
        # write_to_tecplot(cell_data["rho"], f"{self.config.file_config.vtk_path}/tecplot_{str(step).zfill(10)}.dat")


class LBMD3Q27(LBMD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LBMD3Q19(LBMD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LBMD3Q15(LBMD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LBMD3Q13(LBMD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
