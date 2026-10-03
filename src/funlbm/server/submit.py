import json
import os
import re
import shlex
from datetime import datetime

from farlog import getLogger
from funshell import run_shell

from .base import funlbm_cli

logger = getLogger("funlbm")

# 任务名直接拼接进 shell 命令（sbatch 作业名、可执行文件名），
# 只允许字母、数字、下划线和短横线，避免命令注入。
_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@funlbm_cli.command()
def submit(config: str = "./config.json") -> None:
    """提交 Slurm、C++ 或 funlbm 任务；找不到任务时以失败退出。

    根据当前目录下的文件自动判断任务类型并提交：
    存在 ``config.slurm`` 则提交到 Slurm 算力平台；存在 ``main.cpp`` 则本地编译运行；
    否则在 `config` 指向的配置文件存在时，以 `funlbm run` 本地后台运行。

    Args:
        config: funlbm 任务的配置文件路径，默认为当前目录下的 config.json。

    Raises:
        FileNotFoundError: 当前目录下找不到可提交的任务（无 config.slurm、
            main.cpp，也没有 `config` 指向的配置文件）时抛出。
        ValueError: 交互输入的任务名包含非法字符（仅允许字母、数字、
            下划线、短横线）时抛出。
    """
    task_dir = os.path.join(
        os.path.expanduser("~"), "workbench", datetime.now().strftime("%Y%m%d%H%M%S")
    )
    # 对路径做 shell 转义，避免 HOME 等路径含空格或特殊字符时命令被截断/注入。
    task_dir_q = shlex.quote(task_dir)
    logger.info(f"任务主目录：{task_dir}")
    os.makedirs(task_dir, exist_ok=True)
    logger.info(f"复制文件到任务主目录：{task_dir}")
    run_shell(
        f"cp -r *.cpp *.h *.sh *.slurm *.f90 *.dat *.json {task_dir_q} 2>/dev/null"
    )
    logger.success(f"复制文件到任务主目录：{task_dir}完成")

    task_name = input("请输入任务名字，默认为funlbm:").strip()
    if len(task_name) == 0:
        task_name = "funlbm"
    if not _TASK_NAME_RE.fullmatch(task_name):
        raise ValueError(f"任务名只能包含字母、数字、下划线和短横线：{task_name!r}")

    if os.path.exists("config.slurm"):
        logger.info("检测到config.slurm文件，提交到算力平台")
        config_data = open(f"{task_dir}/config.slurm").read().split("\n")
        config_data = [
            f"#SBATCH -J {task_name}" if x.startswith("#SBATCH -J") else x
            for x in config_data
        ]
        with open(f"{task_dir}/config.slurm", "w") as fw:
            fw.write("\n".join(config_data))

        run_shell(f"cd {task_dir_q} && sbatch config.slurm")
        return

    if os.path.exists("main.cpp"):
        logger.info("检测到main.cpp文件，当做C++任务本地运行")
        logger.info("编译main.cpp")
        run_shell(f"cd {task_dir_q} && g++ main.cpp -o {task_name}-task.app")
        logger.info("编译完成，开始执行。")
        run_shell(
            f"""cd {task_dir_q} && nohup ./{task_name}-task.app > output.log 2>&1 &"""
        )
        with open(f"{task_dir}/task.json", "w") as fw:
            fw.write(json.dumps({"task_name": task_name}, indent=2))
        return

    if os.path.exists(config):
        logger.info("检测到config.json文件，当做funlbm任务本地运行")
        # 显式传递 --config，避免非默认配置文件名（如 job.json）复制到任务目录后，
        # 后台的 `funlbm run` 仍按默认 ./config.json 查找而启动失败。
        config_name_q = shlex.quote(os.path.basename(config))
        run_shell(
            f"""cd {task_dir_q} && nohup funlbm run --config {config_name_q} > output.log 2>&1 &"""
        )
        return

    raise FileNotFoundError("找不到需要提交的任务")
