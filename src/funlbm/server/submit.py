import json
import os
from datetime import datetime

from funbuild.shell import run_shell
from funutil import getLogger

from .base import funlbm_cli

logger = getLogger("funlbm")


@funlbm_cli.command()
def submit(config="./config.json"):
    task_dir = os.path.join(
        os.path.expanduser("~"), "workbench", datetime.now().strftime("%Y%m%d%H%M%S")
    )
    logger.info(f"任务主目录：{task_dir}")
    os.makedirs(task_dir, exist_ok=True)
    logger.info(f"复制文件到任务主目录：{task_dir}")
    run_shell(f"cp -r *.cpp *.h *.sh *.slurm *.f90 *.dat *.json {task_dir} 2>/dev/null")
    logger.success(f"复制文件到任务主目录：{task_dir}完成")

    task_name = input("请输入任务名字，默认为funlbm:")
    if task_name is None or len(task_name) == 0:
        task_name = "funlbm"

    if os.path.exists("config.slurm"):
        logger.info("检测到config.slurm文件，提交到算力平台")
        config_data = open(f"{task_dir}/config.slurm").read().split("\n")
        config_data = [
            f"#SBATCH -J {task_name}" if x.startswith("#SBATCH -J") else x
            for x in config_data
        ]
        with open(f"{task_dir}/config.slurm", "w") as fw:
            fw.write("\n".join(config_data))

        run_shell(f"cd {task_dir} && sbatch config.slurm")
        return

    if os.path.exists("main.cpp"):
        logger.info("检测到main.cpp文件，当做C++任务本地运行")
        logger.info("编译main.cpp")
        run_shell(f"cd {task_dir} && g++ main.cpp -o {task_name}-task.app")
        logger.info("编译完成，开始执行。")
        run_shell(
            f"""cd {task_dir} && nohup ./{task_name}-task.app > output.log 2>&1 &"""
        )
        with open(f"{task_dir}/task.json", "w") as fw:
            fw.write(json.dumps({"task_name": task_name}, indent=2))
        return

    if os.path.exists(config):
        logger.info("检测到config.json文件，当做funlbm任务本地运行")
        run_shell(f"""cd {task_dir} && nohup funlbm run > output.log 2>&1 &""")
        return

    logger.error("找不到需要提交的任务")
