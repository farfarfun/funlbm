import h5py
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

    def dump_file(self, group: h5py.Group = None, vals=None, *args, **kwargs) -> None:
        """把对象状态写入 HDF5 分组，默认空实现，由子类按需覆盖。

        Args:
            group: 目标 HDF5 分组，为 None 时不做任何事。
            vals: 需要写入的字段名列表，为 None 时由子类决定默认写入哪些字段。
        """

    def load_file(self, group: h5py.Group = None, vals=None, *args, **kwargs) -> None:
        """从 HDF5 分组恢复对象状态，默认空实现，由子类按需覆盖。

        Args:
            group: 来源 HDF5 分组，为 None 时不做任何事。
            vals: 需要读取的字段名列表，为 None 时由子类决定默认读取哪些字段。
        """

    def dump_dataset(self, group: h5py.Group, name: str, data, *args, **kwargs) -> None:
        """把单个张量以压缩数据集的形式写入 HDF5 分组。

        Args:
            group: 目标 HDF5 分组。
            name: 数据集名称。
            data: 待写入的张量，会先搬到 CPU 再转换为 numpy 数组。
        """
        group.create_dataset(
            name, data=data.cpu().numpy(), compression="gzip", compression_opts=9
        )
        logger.debug(f"dump {name} success.")
        # logger.success(f"dump {name} success.")

    def load_dataset(self, group: h5py.Group, name, *args, **kwargs) -> torch.Tensor:
        """从 HDF5 分组读取单个数据集并转换为当前 device 上的张量。

        Args:
            group: 来源 HDF5 分组。
            name: 数据集名称。

        Returns:
            读取到的张量，位于 `self.device` 上，dtype 为 float32。
        """
        data = torch.tensor(group.get(name)[:], device=self.device, dtype=torch.float32)
        logger.success(f"load {name} success.")
        return data
