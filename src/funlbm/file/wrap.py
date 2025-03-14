import os

from funtable.kv import SQLiteStore

from funlbm.config.base import BaseConfig


class SaveVal:
    def __init__(
        self, per_step=10000000, flow_val=None, particle_val=None, *args, **kwargs
    ):
        self.per_step = per_step
        self.flow_val = flow_val or []
        self.particle_val = particle_val or []


class FileConfig(BaseConfig):
    def __init__(
        self,
        cache_dir: str = "./data",
        custom=None,
        checkpoint=None,
        constant=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache_dir: str = cache_dir
        # 自定义变量
        custom = custom or {
            "per_step": 10,
            "flow_val": ["u", "v", "w", "rou"],
            "particle_val": ["lu", "lrou", "lF"],
        }
        # checkpoint必须有的基础变量
        checkpoint = checkpoint or {
            "per_step": 100,
            # "flow_val": ["u", "v", "w"],
            # "particle_val": ["lu", "lrou", "lF"],
        }
        # 静态不变的变量
        constant = constant or {
            "per_step": 1,
            "flow_val": ["u", "v", "w"],
            "particle_val": ["lx", "lm", "lu_s"],
        }
        self.custom = SaveVal(**custom)
        self.checkpoint = SaveVal(**checkpoint)
        self.constant = SaveVal(**constant)


class FileWrap:
    def __init__(self, config: FileConfig, *args, **kwargs):
        self.config: FileConfig = config
        os.makedirs(self.config.cache_dir, exist_ok=True)

        self.db_store = SQLiteStore(os.path.join(self.config.cache_dir, "track.db"))
        self.db_store.create_kv_table("flow")
        self.db_store.create_kkv_table("particle")
        self.track_flow = self.db_store.get_table("flow")
        self.track_particle = self.db_store.get_table("particle")

    @property
    def checkpoint_dir(self):
        checkpoint_dir = os.path.join(self.config.cache_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    @property
    def custom_dir(self):
        custom_dir = os.path.join(self.config.cache_dir, "custom")
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir

    def checkpoint_path(self, step):
        return f"{self.checkpoint_dir}/checkpoint-{str(step).zfill(10)}.h5"

    def lasted_checkpoint_path(self):
        paths = [
            os.path.join(self.checkpoint_dir, file)
            for file in os.listdir(self.checkpoint_dir)
        ]
        paths = sorted(paths, key=lambda x: x)
        return paths[-1] if len(paths) > 0 else None

    def custom_path(self, step):
        return f"{self.custom_dir}/custom-{str(step).zfill(10)}.h5"

    def lasted_custom_path(self):
        paths = [
            os.path.join(self.custom_dir, file) for file in os.listdir(self.custom_dir)
        ]
        paths = sorted(paths, key=lambda x: x)
        return paths[-1] if len(paths) > 0 else None

    def constant_path(self, *args, **kwargs):
        return f"{self.config.cache_dir}/constant.h5"

    @property
    def config_path(self):
        return f"{self.config.cache_dir}/config.json"
