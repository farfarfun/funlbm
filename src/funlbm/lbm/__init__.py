import os.path

from funlbm.util import logger

from .base import Config, LBMBase
from .lbm3d import LBMD3, LBMD3Q19

__all__ = [
    "LBMD3",
    "LBMD3Q19",
    "Config",
    "LBMBase",
    "create_lbm",
    "create_lbm_by_checkpoint",
]


def create_lbm(config: str | Config = "./config.json") -> LBMD3:
    """创建三维 LBM 求解器实例

    Args:
        config: 配置文件路径或已加载的 Config 对象，默认为当前目录下的 config.json

    Returns:
        LBMD3: 初始化好配置的三维 LBM 求解器
    """
    real_config: Config = (
        config if isinstance(config, Config) else Config.load_config(config)
    )
    return LBMD3(real_config)


def create_lbm_by_checkpoint(filedir: str) -> LBMD3:
    """从检查点目录恢复 LBM 求解器

    Args:
        filedir: 检查点所在目录，目录下需包含 config.json

    Returns:
        LBMD3: 已加载检查点状态的三维 LBM 求解器

    Raises:
        FileNotFoundError: 目录下找不到 config.json 时抛出
    """
    config_path = os.path.join(filedir, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config not found,config path='{config_path}'")
        raise FileNotFoundError(f"config not found,config path='{config_path}'")

    config = Config.load_config(config_path)
    config.file_config.cache_dir = filedir
    lbm = create_lbm(config)
    lbm.init()
    lbm.load_checkpoint(filedir)
    return lbm
