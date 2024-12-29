import json
import os
from enum import Enum

from funutil import deep_get


class BoundaryCondition(Enum):
    PERIODICAL = 11000
    WALL = 1200
    WALL_WITH_SPEED = 1201
    FAR_FIELD = 1300
    NON_EQUILIBRIUM = 1400
    NON_EQUILIBRIUM_EXREAPOLATION = 1500
    FULL_DEVELOPMENT = 1600

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

    def from_file(self, path):
        self.from_json(json.loads(open(path).read()))
        return self

    def from_json(self, config_json: dict, *args, **kwargs):
        self.expand.update(kwargs)
        self.expand.update(config_json)
        self._from_json(config_json, *args, **kwargs)
        return self

    def get(self, key, default=None):
        return deep_get(self.expand, key) or default

    def to_json(self):
        return self.expand


class Boundary(BaseConfig):
    def __init__(
        self, condition: BoundaryCondition = BoundaryCondition.WALL, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.condition: BoundaryCondition = condition
        self.poiseuille = None

    def is_condition(self, condition: BoundaryCondition):
        return self.condition == condition

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.condition = BoundaryCondition.find(deep_get(config_json, "code") or "WALL")
        self.poiseuille = deep_get(config_json, "poiseuille")


class BoundaryConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = Boundary(BoundaryCondition.WALL)
        self.output = Boundary(BoundaryCondition.WALL)

        self.back = Boundary(BoundaryCondition.WALL)
        self.forward = Boundary(BoundaryCondition.WALL)
        self.bottom = Boundary(BoundaryCondition.WALL)
        self.top = Boundary(BoundaryCondition.WALL)

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.input.from_json(deep_get(config_json, "input") or {})
        self.output.from_json(deep_get(config_json, "output") or {})

        self.back.from_json(deep_get(config_json, "back") or {})
        self.forward.from_json(deep_get(config_json, "forward") or {})
        self.bottom.from_json(deep_get(config_json, "bottom") or {})
        self.top.from_json(deep_get(config_json, "top") or {})


class FileConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = "./data"
        self.per_steps = 100

    @property
    def vtk_path(self):
        path = f"{self.cache_dir}/vtk"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.cache_dir = deep_get(config_json, "cache_dir") or self.cache_dir
        self.per_steps = deep_get(config_json, "per_steps") or self.per_steps
