import os

from funtable.kv import SQLiteStore

from funlbm.config.base import BaseConfig


class SaveVal:
    """保存项配置，定义保存频率以及流场和粒子字段。

    参数:
        per_step: 两次保存之间的模拟步数。
        flow_val: 需要保存的流场字段。
        particle_val: 需要保存的粒子字段。
    """

    def __init__(
        self,
        per_step: int = 10000000,
        flow_val: list[str] | None = None,
        particle_val: list[str] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.per_step = per_step
        self.flow_val = flow_val or []
        self.particle_val = particle_val or []


class FileConfig(BaseConfig):
    """文件输出配置，管理缓存目录和各类保存项。

    参数:
        cache_dir: 输出文件的根目录。
        custom: 自定义输出配置。
        checkpoint: 检查点输出配置。
        constant: 静态数据输出配置。
    """

    def __init__(
        self,
        cache_dir: str = "./data",
        custom: dict[str, object] | None = None,
        checkpoint: dict[str, object] | None = None,
        constant: dict[str, object] | None = None,
        *args: object,
        **kwargs: object,
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
    """文件路径与跟踪数据库的统一访问入口。

    参数:
        config: 文件输出配置。
    """

    def __init__(self, config: FileConfig, *args: object, **kwargs: object) -> None:
        self.config: FileConfig = config
        os.makedirs(self.config.cache_dir, exist_ok=True)

        self.db_store = SQLiteStore(os.path.join(self.config.cache_dir, "track.db"))
        self.db_store.create_kv_table("flow")
        self.db_store.create_kkv_table("particle")
        self.track_flow = self.db_store.get_table("flow")
        self.track_particle = self.db_store.get_table("particle")

    @property
    def checkpoint_dir(self) -> str:
        """返回检查点目录，并在目录不存在时创建它。"""
        checkpoint_dir = os.path.join(self.config.cache_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    @property
    def custom_dir(self) -> str:
        """返回自定义输出目录，并在目录不存在时创建它。"""
        custom_dir = os.path.join(self.config.cache_dir, "custom")
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir

    def checkpoint_path(self, step: int) -> str:
        """根据模拟步数返回检查点文件路径。"""
        return f"{self.checkpoint_dir}/checkpoint-{str(step).zfill(10)}.h5"

    def lasted_checkpoint_path(self) -> str | None:
        """返回最新检查点路径；没有检查点时返回 ``None``。"""
        paths = [
            os.path.join(self.checkpoint_dir, file)
            for file in os.listdir(self.checkpoint_dir)
        ]
        paths = sorted(paths, key=lambda x: x)
        return paths[-1] if len(paths) > 0 else None

    def custom_path(self, step: int) -> str:
        """根据模拟步数返回自定义输出文件路径。"""
        return f"{self.custom_dir}/custom-{str(step).zfill(10)}.h5"

    def lasted_custom_path(self) -> str | None:
        """返回最新自定义输出路径；没有输出时返回 ``None``。"""
        paths = [
            os.path.join(self.custom_dir, file) for file in os.listdir(self.custom_dir)
        ]
        paths = sorted(paths, key=lambda x: x)
        return paths[-1] if len(paths) > 0 else None

    def constant_path(self, *args: object, **kwargs: object) -> str:
        """返回静态数据文件路径。"""
        return f"{self.config.cache_dir}/constant.h5"

    @property
    def config_path(self) -> str:
        """返回配置文件路径。"""
        return f"{self.config.cache_dir}/config.json"
