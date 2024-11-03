import funutil
import numpy as np
import torch
from funutil import run_timer
from funvtk.hl import gridToVTK, pointsToVTK

from funlbm.config import Config
from funlbm.flow import FlowD3
from funlbm.particle import Ellipsoid

from ..file.tecplot.dump import write_to_tecplot
from .base import LBMBase

logger = funutil.getLogger("funlbm")


class LBMD3(LBMBase):
    def __init__(self, config: Config, *args, **kwargs):
        flow = FlowD3(config=config.flow_config, *args, **kwargs)
        particles = [Ellipsoid(config=con) for con in config.particles]
        super(LBMD3, self).__init__(flow=flow, config=config, particles=particles, *args, **kwargs)

    def init(self):
        # 初始化流程
        # 初始化边界条件
        self.flow.init()
        # 初始化颗粒

        # self = self.particles[0]
        self.flow.cul_equ()

        for particle in self.particles:
            particle.init()

    @run_timer
    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        for particle in self.particles:
            rl = np.array(np.floor(particle.lx.to("cpu").numpy()) - (n - 1) * h, dtype=int)
            rl[rl < 0] = 0
            rr = rl + (2 * n - 1) * h + 1

            lu = torch.zeros(particle.lu.shape, device=self.device)
            for index, lar in enumerate(particle.lx):
                i0, i1 = rl[index], rr[index]
                tmp = self.flow.x[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] - lar
                tmp = (1 + torch.cos(torch.abs(tmp / h * np.pi / 2 / h))) / 4 / h
                tmp = torch.prod(tmp, dim=-1, keepdim=True)

                lu[index, :] = torch.sum(self.flow.u[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] * tmp)
                particle.lrou[index, :] = torch.sum(self.flow.rou[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] * tmp)
            u_theta = 0

            particle.lu[:, :] = (
                particle.cu + torch.linalg.cross(particle.cw.unsqueeze(0), particle.lx - particle.cx, dim=-1) + u_theta
            )
            particle.lF = particle.lrou * (particle.lu - lu)

            # TODO 力矩的公式到底是r×F，F×r
            # particle.lT = np.cross(particle.lF, particle.lx - particle.cx)
            particle.lT = torch.cross(particle.lx - particle.cx, particle.lF, dim=-1)

    @run_timer
    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        for particle in self.particles:
            rl = np.array(np.floor(particle.lx.to("cpu").numpy()) - (n - 1) * h, dtype=int)
            rl[rl < 0] = 0
            rr = rl + (2 * n - 1) * h + 1
            for index, lar in enumerate(particle.lx):
                i0, i1 = rl[index], rr[index]
                tmp = self.flow.x[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] - lar
                tmp = (1 + torch.cos(torch.abs(tmp / h * np.pi / 2 / h))) / 4 / h
                tmp = torch.prod(tmp, dim=-1, keepdim=True)
                tmp = tmp * particle.lF[index, :] * particle.lm[index]
                self.flow.FOL[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] = tmp

    @run_timer
    def particle_to_wall(self, *args, **kwargs):
        k0 = 300
        for particle in self.particles:
            n = torch.tensor(
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                device=self.device,
                dtype=torch.float32,
            )
            xt = torch.concatenate(
                [
                    torch.zeros(self.config.flow_config.size.shape, device=self.device),
                    torch.tensor(self.config.flow_config.size, device=self.device, dtype=torch.float32),
                ]
            )
            xi = torch.concatenate([torch.argmin(particle.lx, dim=0), torch.argmax(particle.lx, dim=0)])

            d = k0 * (1 - abs(particle.lx[xi][[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]] - xt) / (2 * self.config.dx))
            d[d < 0] = 0
            if d.max() == 0:
                continue
            particle.lF[xi] = particle.lF[xi] + torch.multiply(n, d.unsqueeze(1))

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

        cell_data = {"rho": self.flow.rou.to("cpu").numpy()[:, :, :, 0], "p": self.flow.p.to("cpu").numpy()[:, :, :, 0]}

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
        write_to_tecplot(cell_data["rho"], f"{self.config.file_config.vtk_path}/tecplot_{str(step).zfill(10)}.dat")


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
