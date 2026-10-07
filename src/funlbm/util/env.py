import os
from multiprocessing import cpu_count

import torch


def set_cpu() -> None:
    """将数值计算库的线程数设置为当前机器的 CPU 核数。

    Returns:
        无返回值；设置 OpenMP、BLAS 和 PyTorch 的线程数。
    """
    cpu_num = cpu_count()  # 这里设置成你想运行的CPU个数
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)
