import torch

from funlbm.util import logger


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


class Worker:
    def __init__(self, device="cpu", *args, **kwargs):
        self.device = device_detect(device)
        self.device = device
