from typing import Dict

import h5py
import torch
from funlog import getLogger
from funutil import run_timer

from funlbm.base import Worker
from funlbm.config.base import BaseConfig
from funlbm.particle.coord import CoordConfig, Coordinate
from funlbm.util import tensor_format

logger = getLogger("funlbm")


class ParticleConfig(BaseConfig):
    def __init__(self, coord=None, type="ellipsoid", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_config: CoordConfig = CoordConfig(**coord or {})
        self.type = type


class Particle(Worker):
    """
    粒子基类

    属性:
        config (ParticleConfig): 粒子配置
        coord (Coordinate): 坐标系对象
        mass (float): 粒子质量
        area (float): 粒子表面积
        I (Tensor): 惯性矩
        rou (float): 粒子密度
        angle (Tensor): 粒子方向
        cx (Tensor): 质心坐标 [x,y,z]
        cr (Tensor): 质心半径 [a,b,b]
        cu (Tensor): 质心速度 [vx,vy,vz]
        cw (Tensor): 质心角速度 [wx,wy,wz]
        cF (Tensor): 质心合外力
        cT (Tensor): 质心合外力矩
        lx (Tensor): 拉格朗日点坐标 [m,i,3]
        lF (Tensor): 拉格朗日点受力 [m,i,3]
        lm (Tensor): 拉格朗日点质量
        lu (Tensor): 拉格朗日点速度 [m,i,3]
        lrou (Tensor): 拉格朗日点密度
    """

    def __init__(self, config: ParticleConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: ParticleConfig = config or ParticleConfig()
        self.coord: Coordinate = Coordinate(config=self.config.coord_config, *args, **kwargs)

        # 颗粒质量[1]
        self.mass = None
        # 颗粒面积[1]
        self.area = None
        # 惯性矩
        self.I = None
        # 颗粒密度[1]
        self.rou = None
        # 颗粒方向[i,j,k]
        self.angle = None

        # 质心坐标[i,j,k]
        self.cx = torch.tensor(self.config.coord_config.center, device=self.device, dtype=torch.float32)
        # 质心半径[a,b,b]
        self.cr = 5 * torch.ones(5, device=self.device, dtype=torch.float32)
        # 质心速度[i,j,k]
        self.cu = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心角速度[i,j,k]
        self.cw = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cF = torch.zeros(3, device=self.device, dtype=torch.float32)
        # 质心合外力
        self.cT = torch.zeros(3, device=self.device, dtype=torch.float32)

        self._lagrange: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        # 拉格朗日点的坐标[m,i,3]
        self.lx: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        self.lu_s: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        # 拉格朗日点上的力[m,i,3]
        self.lF: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        # 拉格朗日点的质量
        self.lm: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        # 拉格朗日点速度[m,i,3]
        self.lu: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)
        # 拉格朗日点速度[m,i,3]
        self.lrou: torch.Tensor = torch.zeros([0], device=self.device, dtype=torch.float32)

    def _init(self, dx=1, *args, **kwargs):
        raise NotImplementedError("还没实现")

    def init(self, *args, **kwargs):
        self.rou = float(self.config.get("rou", 1.0))
        self._init(*args, **kwargs)
        shape = self._lagrange.shape
        self.lx = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lF = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lu = torch.zeros_like(self._lagrange, dtype=torch.float32)
        self.lm = torch.full((shape[0], 1), self.area / shape[0], device=self.device, dtype=torch.float32)
        self.lrou = torch.empty((shape[0], 1), device=self.device, dtype=torch.float32)

        logger.info(f"lagrange shape: {shape}")
        logger.info(f"area is {self.area}")
        logger.info(f"mass is {self.mass}")

    @run_timer
    def update_from_lar(self, dt: float, gl: float = 9.8, rouf: float = 1.0) -> None:
        """
        从拉格朗日坐标更新粒子状态。

        参数:
            dt: 时间步长
            gl: 重力加速度
            rouf: 流体密度

        异常:
            ValueError: 当输入参数无效时抛出
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if rouf <= 0:
            raise ValueError("Fluid density must be positive")

        tmp = (
            # (1 - rouf / self.rou)*
            self.mass * torch.tensor([0, 0, -gl], device=self.device)
        )
        self.cF = torch.sum(-self.lF * self.lm, dim=0) + tmp

        self.cu = self.cu + self.cF / self.mass * dt
        self.cx = self.cx + self.cu * dt

        self.cT = -torch.sum(torch.cross(self.lx - self.cx, self.lF, dim=-1) * self.lm, dim=0)
        self.cw = self.cw + self.cT * dt / self.I
        # TODO 这里是临时不让旋转
        # self.cw = 0 * self.cw

    @run_timer
    def update(self, *args, **kwargs):
        self.coord.update(center=self.cx, w=self.cw)
        self.lx = self.coord.cul_point(self._lagrange)

    def from_json(self):
        pass

    def track(self, step=0, *args, **kwargs) -> Dict:
        return {
            "m": float(self.mass.cpu().numpy()),
            "cu": tensor_format(self.cu),
            "cx": tensor_format(self.cx),
            "cf": tensor_format(self.cF),
            "lF": tensor_format(
                [
                    self.lF.min(),
                    self.lF.mean(),
                    self.lF.max(),
                ]
            ),
            "cw": tensor_format(self.cw),
            "coord": self.coord.to_json(),
        }

    def dump_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or [
            "cu",
            "cx",
            "cF",
            "cw",
            "lx",
            "lu",
            "lF",
            "lrou",
            "lu_s",
            "lm",
            "_lagrange",
            "coord",
        ]
        if "cu" in vals:
            self.dump_dataset(group, "cu", self.cu)
        if "cx" in vals:
            self.dump_dataset(group, "cx", self.cx)
        if "cF" in vals:
            self.dump_dataset(group, "cF", self.cF)
        if "cw" in vals:
            self.dump_dataset(group, "cw", self.cw)
        if "lx" in vals:
            self.dump_dataset(group, "lx", self.lx)
        if "lu" in vals:
            self.dump_dataset(group, "lu", self.lu)
        if "lF" in vals:
            self.dump_dataset(group, "lF", self.lF)
        if "lrou" in vals:
            self.dump_dataset(group, "lrou", self.lrou)
        if "lu_s" in vals:
            self.dump_dataset(group, "lu_s", self.lu_s)
        if "lm" in vals:
            self.dump_dataset(group, "lm", self.lm)
        if "_lagrange" in vals:
            self.dump_dataset(group, "_lagrange", self._lagrange)
        if "coord" in vals:
            self.coord.dump_file(group.create_group("coord"))

    def load_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or group.keys()
        if "lm" in vals:
            self.lm = self.load_dataset(group, "lm")
        if "cu" in vals:
            self.cu = self.load_dataset(group, "cu")
        if "cx" in vals:
            self.cx = self.load_dataset(group, "cx")
        if "cF" in vals:
            self.cF = self.load_dataset(group, "cF")
        if "cw" in vals:
            self.cw = self.load_dataset(group, "cw")
        if "lx" in vals:
            self.lx = self.load_dataset(group, "lx")
        if "lu" in vals:
            self.lu = self.load_dataset(group, "lu")
        if "lF" in vals:
            self.lF = self.load_dataset(group, "lF")
        if "lrou" in vals:
            self.lrou = self.load_dataset(group, "lrou")
        if "lu_s" in vals:
            self.lu_s = self.load_dataset(group, "lu_s")

        if "_lagrange" in vals:
            self._lagrange = self.load_dataset(group, "_lagrange")
        if "coord" in vals:
            self.coord.load_file(group.get("coord"))
