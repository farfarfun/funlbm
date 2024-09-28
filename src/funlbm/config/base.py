import os
from enum import Enum
from typing import List

import numpy as np

from funlbm.utils import deep_get


class BoundaryCondition(Enum):
    PERIODICAL = 11000
    WALL = 1200
    WALL_WITH_SPEED = 1201
    FAR_FIELD = 1300
    NON_EQUILIBRIUM = 1400
    FULL_DEVELOPMENT = 1500

    @staticmethod
    def find(code):
        for value in BoundaryCondition.__iter__():
            if code == value.value or code == str(value.name):
                return value
        return BoundaryCondition.WALL


class BaseConfig(object):
    def __init__(self, *args, **kwargs):
        self.expand = {}
        self.expand.update(kwargs)

    def _from_json(self, config_json: dict, *args, **kwargs):
        pass

    def from_json(self, config_json: dict, *args, **kwargs):
        self.expand.update(kwargs)
        self._from_json(config_json, *args, **kwargs)
        return self

    def get(self, key, default=None):
        value = deep_get(self.expand, key)
        if value is None:
            return default


class Boundary(BaseConfig):
    def __init__(
        self, condition: BoundaryCondition = BoundaryCondition.WALL, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.condition: BoundaryCondition = condition
        self.poiseuille = None
        self.config = {}
        self.config.update(kwargs)

    def get(self, key):
        return self.config[key]

    def is_condition(self, condition: BoundaryCondition):
        return self.condition == condition

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.condition = deep_get(config_json, "code") or self.condition
        self.poiseuille = deep_get(config_json, "poiseuille")
        self.config.update(kwargs)


class BoundaryConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left = Boundary(BoundaryCondition.WALL)
        self.right = Boundary(BoundaryCondition.WALL)
        self.back = Boundary(BoundaryCondition.WALL)
        self.forward = Boundary(BoundaryCondition.WALL)
        self.bottom = Boundary(BoundaryCondition.WALL)
        self.top = Boundary(BoundaryCondition.WALL)

    def _from_json(self, config_json: dict, *args, **kwargs):
        if "left" in config_json.keys():
            self.left = Boundary().from_json(config_json["left"])
        if "right" in config_json.keys():
            self.right = Boundary().from_json(config_json["right"])
        if "back" in config_json.keys():
            self.left = Boundary().from_json(config_json["back"])
        if "forward" in config_json.keys():
            self.left = Boundary().from_json(config_json["forward"])

        if "bottom" in config_json.keys():
            self.left = Boundary().from_json(config_json["bottom"])
        if "top" in config_json.keys():
            self.left = Boundary().from_json(config_json["left"])


class FileConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = "./data"

    @property
    def vtk_path(self):
        path = f"{self.cache_dir}/vtk"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.cache_dir = deep_get(config_json, "cache_dir") or self.cache_dir


class CoordConfig(BaseConfig):
    def __init__(self, alpha=np.pi / 2, beta=0, gamma=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center = [0, 0, 0]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.center = deep_get(config_json, "center") or self.center
        self.alpha = deep_get(config_json, "alpha") or self.alpha
        self.beta = deep_get(config_json, "beta") or self.beta
        self.gamma = deep_get(config_json, "gamma") or self.gamma


class ParticleConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_config: CoordConfig = CoordConfig()

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.coord_config.from_json(deep_get(config_json, "coord"))


class FlowConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        self.size = np.zeros(3)
        self.boundary: BoundaryConfig = None
        super().__init__(*args, **kwargs)

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.size = np.array(
            deep_get(config_json, "size") or [100, 100, 100], dtype=int
        )
        self.boundary = BoundaryConfig().from_json(
            deep_get(config_json, "boundary") or {}
        )


class Config(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = 0.1
        self.dx = 0.1
        self.file_config = FileConfig()
        self.flow_config = FlowConfig()
        self.particles: List[ParticleConfig] = []

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.dt = deep_get(config_json, "dt") or self.dt
        self.dx = deep_get(config_json, "dx") or self.dx
        self.file_config = FileConfig().from_json(deep_get(config_json, "file") or {})
        self.flow_config = FlowConfig().from_json(deep_get(config_json, "flow") or {})
        for config in deep_get(config_json, "particles") or []:
            self.particles.append(ParticleConfig().from_json(config_json=config))
