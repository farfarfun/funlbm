from typing import Any

import numpy as np
import pandas as pd
from funtable.kv import SQLiteKKVTable
from nicegui import ui


def trans(track: dict, particle_id: str = "1") -> pd.DataFrame:
    """将 track 字典（SQLiteKKVTable 中记录的逐步颗粒状态）展开为 DataFrame。

    Args:
        track: `{step: {particle_id: {...状态字段...}}}` 结构的逐步记录。
        particle_id: 要提取的颗粒 id。

    Returns:
        按 step 展开的 DataFrame，包含速度/位置/角速度/角度等列，
        并附加合速度列 `u`。
    """
    res = [
        {
            "step": step,
            "m": track[step][particle_id]["m"],
            "ux": track[step][particle_id]["cu"][0],
            "uy": track[step][particle_id]["cu"][1],
            "uz": track[step][particle_id]["cu"][2],
            "cx": track[step][particle_id]["cx"][0],
            "cy": track[step][particle_id]["cx"][1],
            "cz": track[step][particle_id]["cx"][2],
            "cfx": track[step][particle_id]["cf"][0],
            "cfy": track[step][particle_id]["cf"][1],
            "cfz": track[step][particle_id]["cf"][2],
            "lfx": track[step][particle_id]["lF"][0],
            "lfy": track[step][particle_id]["lF"][1],
            "lfz": track[step][particle_id]["lF"][2],
            "cwx": track[step][particle_id]["cw"][0],
            "cwy": track[step][particle_id]["cw"][1],
            "cwz": track[step][particle_id]["cw"][2],
            "anglex": track[step][particle_id]["coord"]["angle"][0],
            "angley": track[step][particle_id]["coord"]["angle"][1],
            "anglez": track[step][particle_id]["coord"]["angle"][2],
            "centerx": track[step][particle_id]["coord"]["center"][0],
            "centery": track[step][particle_id]["coord"]["center"][1],
            "centerz": track[step][particle_id]["coord"]["center"][2],
        }
        for step in track
    ]
    df = pd.DataFrame(res)
    df["u"] = np.sqrt(df["ux"] ** 2 + df["uy"] ** 2 + df["uz"] ** 2)
    return df


def plt_u(
    df: pd.DataFrame,
    keys: list[str],
    title: str = "折线图",
    total: int | None = None,
) -> Any:
    """根据跟踪数据创建速度、位置等指标的折线图。

    Args:
        df: 包含 ``step`` 和待绘制数值列的数据表。
        keys: 要绘制的列名列表。
        title: 图表标题。
        total: 保留的最大数据量，当前由调用方负责切片。

    Returns:
        NiceGUI 创建的 ECharts 图表对象。
    """
    chart = ui.echart(
        {
            "title": {"text": title},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [key for key in keys]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {"type": "category", "data": df["step"].tolist()},
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": key,
                    "data": df[key].tolist(),
                    "type": "line",
                    "smooth": True,
                }
                for key in keys
            ],
        }
    )
    return chart


def plt_job(
    job_id: str = "13546157",
    title: str | None = None,
    total: int | None = None,
    step: int = 100,
    home: str = "/Users/bingtao/data/scnet",
) -> None:
    """从任务目录读取跟踪数据库并渲染颗粒状态图表。

    Args:
        job_id: 任务目录名称。
        title: 页面标题；未指定时使用任务编号。
        total: 最多展示的记录数量。
        step: 绘图时的记录采样间隔。
        home: 保存所有任务目录的根路径。

    Returns:
        无返回值，图表直接添加到当前 NiceGUI 页面。
    """
    path = f"{home}/{job_id}"
    track_path = f"{path}/data/track.db"
    particle_table = SQLiteKKVTable(db_path=track_path, table_name="particle")
    track = particle_table.list_all()

    df0 = trans(track)
    total = total or len(df0)
    df0 = df0[:total:step]
    with ui.card():
        ui.label(title or f"任务 {job_id}").classes("text-h6 text-amber-9 q-mb-md")
        with ui.row().classes("w-full h-full"):
            with ui.card().classes("w-2/5 h-full"):
                plt_u(df0, ["u", "ux", "uy", "uz"], "速度")
            with ui.card().classes("w-2/5 h-full"):
                plt_u(df0, ["cwx", "cwy", "cwz"], "角速度")
            with ui.card().classes("w-2/5 h-full"):
                plt_u(df0, ["cx", "cy", "cz"], "位置")
            with ui.card().classes("w-2/5 h-full"):
                plt_u(df0, ["anglex", "angley", "anglez"], "角度")
