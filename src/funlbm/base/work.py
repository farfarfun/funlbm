import torch

from funlbm.util import logger


def device_detect(device=None):
    """检测并返回可用的计算设备

    按优先级顺序检测:CUDA > MPS > CPU

    Args:
        device: 指定的设备名称,如果为None则自动检测

    Returns:
        torch.device: 返回可用的计算设备(cuda/mps/cpu)
    """
    if device is not None and device != "auto":
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


class Worker:
    """基础工作类

    提供设备检测和管理功能

    Args:
        device: 计算设备名称,默认为"cpu"
    """

    def __init__(self, device="cpu", *args, **kwargs):
        self.device = device_detect(device)
        logger.info(f"init {type(self).__name__} with device={self.device}")
