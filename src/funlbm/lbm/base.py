import numpy as np
from funvtk.hl import gridToVTK, pointsToVTK
from tqdm import tqdm

from funlbm.config import Config
from funlbm.flow.flow3d import Flow3D
from funlbm.particle import Ellipsoid


class Solver(object):
    def __init__(self, config: Config, *args, **kwargs):
        self.config = config
        self.flow = Flow3D(config=config.flow_config)
        self.particles = [Ellipsoid(config=con) for con in config.particles]

    def run(self):
        self.init()
        total = int(9000 / self.config.dt)
        pbar = tqdm(range(total))
        pbar = range(total)
        for step in pbar:
            self.step(step)
            # print(f"{step}"
            #       f"\t{np.max(self.flow.f):8f}"
            #       f"\t{np.mean(self.flow.u[1, 1:-1, 1:-1, :]):8f}"
            #       f"\t{np.mean(self.flow.u[10, 1:-1, 1:-1, :]):8f}"
            #       f"\t{np.sum(self.flow.f):8f}"
            #       f"\t{self.particles[0].cx}"
            #       )

    def init(self):
        # 初始化流程
        # 初始化边界条件
        self.flow.init()
        # 初始化颗粒

        # self = self.particles[0]
        self.flow.cul_equ(tau=1)

        for particle in self.particles:
            particle.init()

    def flow_to_lagrange(self, n=2, h=1):
        for particle in self.particles:
            rl = np.array(np.floor(particle.lx) - (n - 1) * h, dtype=int)
            rl[rl < 0] = 0
            rr = rl + (2 * n - 1) * h + 1

            lu = np.zeros(particle.lu.shape)
            for index, lar in enumerate(particle.lx):
                i0, i1 = rl[index], rr[index]
                tmp = self.flow.x[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] - lar
                tmp = (1 + np.cos(np.abs(tmp / h * np.pi / 2 / h))) / 4 / h
                tmp = np.prod(tmp, axis=-1, keepdims=True)

                lu[index, :] = np.sum(
                    self.flow.u[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] * tmp
                )
                particle.lrou[index, :] = np.sum(
                    self.flow.rou[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] * tmp
                )
                # print(lar, i1)
            u_theta = 0

            particle.lu[:, :] = (
                particle.cu + np.cross(particle.cw, particle.lx - particle.cx) + u_theta
            )
            particle.lF = particle.lrou * (particle.lu - lu)

            # TODO 力矩的公式到底是r×F，F×r
            # particle.lT = np.cross(particle.lF, particle.lx - particle.cx)
            particle.lT = np.cross(particle.lx - particle.cx, particle.lF)

    def lagrange_to_flow(self, n=2, h=1):
        for particle in self.particles:
            rl = np.array(np.floor(particle.lx) - (n - 1) * h, dtype=int)
            rl[rl < 0] = 0
            rr = rl + (2 * n - 1) * h + 1
            for index, lar in enumerate(particle.lx):
                i0, i1 = rl[index], rr[index]
                tmp = self.flow.x[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] - lar
                tmp = (1 + np.cos(np.abs(tmp / h * np.pi / 2 / h))) / 4 / h
                tmp = np.prod(tmp, axis=-1, keepdims=True)
                tmp = tmp * particle.lF[index, :] * particle.lm[index]
                self.flow.FOL[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2], :] = tmp

    def particle_to_wall(self):
        k0 = 300
        for particle in self.particles:
            n = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
            )
            xt = np.concatenate(
                [
                    np.zeros(self.config.flow_config.size.shape),
                    self.config.flow_config.size,
                ]
            )
            xi = np.concatenate(
                [np.argmin(particle.lx, axis=0), np.argmax(particle.lx, axis=0)]
            )

            d = k0 * (
                1
                - abs(particle.lx[xi][[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]] - xt)
                / (2 * self.config.dx)
            )
            d[d < 0] = 0
            if d.max() == 0:
                continue
            print(xi, d)
            particle.lF[xi] = particle.lF[xi] + np.multiply(n, np.expand_dims(d, 1))

    def step(self, step):
        # 流场碰撞
        self.flow.cul_equ()
        # 流场迁移
        self.flow.f_stream()
        # 流场计算-速度&密度
        self.flow.update_u_rou()

        # 浸没计算-流场->拉格朗日点
        self.flow_to_lagrange()

        self.particle_to_wall()

        # 浸没计算-拉格朗日点->颗粒
        [particle.update_from_lar(dt=self.config.dt) for particle in self.particles]

        # 浸没计算-拉格朗日点->流场
        self.lagrange_to_flow()

        # 流场二次碰撞
        self.flow.cul_equ2()
        # 流场计算-速度&密度
        self.flow.update_u_rou()

        # 颗粒->拉格朗日点
        [particle.update() for particle in self.particles]
        [print(particle.to_str()) for particle in self.particles]
        if step % 10 == 0 and step > 10:
            self.save(step)

    def save(self, step=10):

        shape = self.flow.u.shape
        xf, yf, zf = np.meshgrid(
            range(shape[0]),
            range(shape[1]),
            range(shape[2]),
            indexing="ij",
            sparse=False,
        )

        point_data = {
            "u": (
                self.flow.u[:, :, :, 0],
                self.flow.u[:, :, :, 1],
                self.flow.u[:, :, :, 2],
            ),
            # "rou": (self.rou[:, :, :, 0], self.rou[:, :, :, 0], self.rou[:, :, :, 0]),
        }

        gridToVTK(
            f"{self.config.file_config.vtk_path}/flow_" + str(step).zfill(10),
            xf,
            yf,
            zf,
            pointData=point_data,
        )

        for i, particle in enumerate(self.particles):
            xf, yf, zf = particle.lx[:, 0], particle.lx[:, 1], particle.lx[:, 2]
            data = {
                # "u": self.lu
                "u": particle.lu[:, 0],  # , self.lu[:, 1], self.lu[:, 2])
                # "u": (self.lu[:, 0], self.lu[:, 1], self.lu[:, 2])
            }
            fname = f"{self.config.file_config.vtk_path}/particle_{str(i).zfill(3)}_{str(step).zfill(10)}"
            pointsToVTK(fname, xf, yf, zf, data=data)
