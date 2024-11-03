import funutil
import torch
from tqdm import tqdm

from funlbm.config import Config
from funlbm.flow import FlowD3

logger = funutil.getLogger("funlbm")


def device_detect(device=None):
    if device is not None:
        return device
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except AttributeError:
        logger.error("No cuda available")
    try:
        if torch.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        logger.error("No mps available")
    return torch.device("cpu")


class LBMBase(object):
    def __init__(self, flow: FlowD3, config: Config, device=None, particles=None, *args, **kwargs):
        self.flow = flow
        self.config = config
        self.particles = particles
        self.device = device_detect(device)
        logger.info(f"device:={self.device}")

    def run(self, max_steps=10000, *args, **kwargs):
        self.init()
        total = int(max_steps / self.config.dt)
        pbar = tqdm(range(total))
        # pbar = range(total)
        for step in pbar:
            self.run_step(step=step)
            # print(
            #     f"{step}" f"\tf_max={np.max(self.flow.f.numpy()):8f}" f"\tu_max={np.max(self.flow.u.numpy()):8f}"
            #     # f"\t{self.particles[0].cx}"
            # )

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def run_step(self, step, *args, **kwargs):
        # 流场碰撞
        self.flow.cul_equ(step=step)
        # 流场迁移
        self.flow.f_stream()
        # 流场计算-速度&密度
        self.flow.update_u_rou(step=step)

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
        self.save(step)

    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def particle_to_wall(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, step=10, *args, **kwargs):
        raise NotImplementedError()
