import os.path
from typing import Union

from funlbm.util import logger

from .base import Config, LBMBase
from .lbm3d import LBMD3, LBMD3Q19

__all__ = ["LBMBase", "LBMD3", "LBMD3Q19", "Config"]


def create_lbm(config: Union[str, Config] = "./config.json"):
    config: Config = (
        config if isinstance(config, Config) else Config.load_config(config)
    )

    return LBMD3(config)


def create_lbm_by_checkpoint(filedir):
    config_path = os.path.join(filedir, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config not found,config path='{config_path}'")
        raise Exception(f"config not found,config path='{config_path}'")

    config = Config.load_config(config_path)
    config.file_config.cache_dir = filedir
    lbm = create_lbm(config)
    lbm.init()
    lbm.load_checkpoint(filedir)
    return lbm
