"""funlbm 冒烟测试与核心路径回归测试。

funlbm 是基于 numpy/scipy/torch/h5py 的三维格子玻尔兹曼方法（LBM）数值模拟包。
这些测试覆盖：包能正常安装导入、公开配置类的正常路径与边界、CLI 入口能正常工作，
以及一个使用极小网格（6x6x6，无颗粒，单步）的端到端 LBM 求解流程，
用来在不引入昂贵计算的前提下验证核心模拟链路（flow -> particle -> checkpoint）没有回归。
"""

import json
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Basic package import
# ---------------------------------------------------------------------------


def test_import_top_level_package():
    import funlbm  # noqa: F401


def test_import_funlbm_config():
    import funlbm.config
    import funlbm.config.base  # noqa: F401


def test_import_funlbm_file():
    import funlbm.file  # noqa: F401


@pytest.mark.parametrize(
    "module_name",
    [
        "funlbm.util",
        "funlbm.base",
        "funlbm.flow",
        "funlbm.particle",
        "funlbm.lbm",
        "funlbm.server",
    ],
)
def test_modules_import_cleanly(module_name):
    import importlib

    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# funlbm.config -- BoundaryCondition / Boundary / BaseConfig
# ---------------------------------------------------------------------------


def test_boundary_condition_find_by_int_and_name():
    from funlbm.config import BoundaryCondition

    assert BoundaryCondition.find(1200) is BoundaryCondition.WALL
    assert BoundaryCondition.find("PERIODICAL") is BoundaryCondition.PERIODICAL
    assert BoundaryCondition.find(11000) is BoundaryCondition.PERIODICAL


def test_boundary_condition_find_unknown_falls_back_to_wall():
    from funlbm.config import BoundaryCondition

    # Unknown code/name should not raise, defaults to WALL.
    assert BoundaryCondition.find("not-a-real-condition") is BoundaryCondition.WALL
    assert BoundaryCondition.find(-1) is BoundaryCondition.WALL


def test_boundary_default_condition_is_wall():
    from funlbm.config import Boundary, BoundaryCondition

    b = Boundary()
    assert b.condition is BoundaryCondition.WALL
    assert b.is_condition(BoundaryCondition.WALL) is True
    assert b.is_condition(BoundaryCondition.PERIODICAL) is False


def test_boundary_with_explicit_periodical_code():
    from funlbm.config import Boundary, BoundaryCondition

    b = Boundary(code="PERIODICAL")
    assert b.is_condition(BoundaryCondition.PERIODICAL) is True


def test_boundary_config_with_explicit_empty_faces():
    from funlbm.config.base import BoundaryConfig

    cfg = BoundaryConfig(input={}, output={}, back={})
    for face in ("input", "output", "back", "forward", "bottom", "top"):
        boundary = getattr(cfg, face)
        assert boundary.condition.name == "WALL"


def test_boundary_config_default_construction():
    """回归测试：BoundaryConfig() 全部使用默认参数时不应报错。

    此前 input/output/back 在 `**` 展开时缺少 `or {}` 兜底（forward/bottom/top
    有），传 None 会直接 TypeError。已在 funlbm.config.base 修复。
    """
    from funlbm.config.base import BoundaryConfig

    cfg = BoundaryConfig()
    for face in ("input", "output", "back", "forward", "bottom", "top"):
        boundary = getattr(cfg, face)
        assert boundary.condition.name == "WALL"


def test_base_config_json_roundtrip():
    from funlbm.config.base import BaseConfig

    cfg = BaseConfig(a=1)
    cfg.from_json({"x": 1, "y": {"z": 2}})

    assert cfg.exists("x") is True
    assert cfg.exists("missing") is False
    assert cfg.get("x") == 1
    assert cfg.get("missing", "default") == "default"
    assert cfg.to_json() == {"a": 1, "x": 1, "y": {"z": 2}}


def test_base_config_from_file(tmp_path):
    from funlbm.config.base import BaseConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"foo": "bar"}))

    cfg = BaseConfig().from_file(str(config_path))
    assert cfg.get("foo") == "bar"


# ---------------------------------------------------------------------------
# funlbm.lbm.base -- Config 顶层配置的正常路径与边界（file/flow/particles 缺省）
# ---------------------------------------------------------------------------


def test_lbm_config_all_defaults():
    """回归测试：Config() 全部使用默认参数时不应报错。

    此前 file/flow 在 `**` 展开时没有 `or {}` 兜底，传 None 会直接 TypeError；
    particles=None 时 `for config in particles` 也会 TypeError。均已修复。
    """
    from funlbm.lbm.base import Config

    cfg = Config()
    assert cfg.file_config.cache_dir == "./data"
    assert cfg.flow_config.param_type == "D3Q19"
    assert cfg.particles == []


# ---------------------------------------------------------------------------
# funlbm.file -- FileConfig / FileWrap (real but tiny filesystem I/O)
# ---------------------------------------------------------------------------


def test_file_config_defaults():
    from funlbm.file import FileConfig

    cfg = FileConfig()
    assert cfg.cache_dir == "./data"
    assert cfg.custom.per_step == 10
    assert cfg.checkpoint.per_step == 100
    assert cfg.constant.per_step == 1


def test_file_wrap_creates_dirs_and_paths(tmp_path):
    """FileWrap does real (but tiny/local) filesystem + sqlite setup, so we
    point it at pytest's tmp_path instead of mocking -- it's cheap and self
    contained (no network, no large data)."""
    from funlbm.file import FileConfig, FileWrap

    cfg = FileConfig(cache_dir=str(tmp_path))
    wrap = FileWrap(cfg)

    assert wrap.checkpoint_dir == str(tmp_path / "checkpoint")
    assert wrap.custom_dir == str(tmp_path / "custom")
    assert (tmp_path / "checkpoint").is_dir()
    assert (tmp_path / "custom").is_dir()
    assert wrap.checkpoint_path(7).endswith("checkpoint-0000000007.h5")
    assert wrap.custom_path(7).endswith("custom-0000000007.h5")
    # No checkpoints/custom files written yet -> "latest" lookups are None.
    assert wrap.lasted_checkpoint_path() is None
    assert wrap.lasted_custom_path() is None


# ---------------------------------------------------------------------------
# CLI 入口 -- funlbm.server (submit/update 已从 funbuild.shell 迁移到 funshell)
# ---------------------------------------------------------------------------


def test_cli_entry_point_help():
    """funlbm 控制台脚本入口 (funlbm.server:funlbm) 应能正常启动并打印帮助。"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from funlbm.server import funlbm; sys.argv=['funlbm', '--help']; funlbm()",
        ],
        capture_output=True,
        text=True,
        env={"COLUMNS": "200"},
    )
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "submit" in result.stdout
    assert "update" in result.stdout


# ---------------------------------------------------------------------------
# 端到端最小 LBM 求解流程（极小网格 6x6x6，无颗粒，单步），验证核心链路无回归
# ---------------------------------------------------------------------------


def test_lbm_end_to_end_single_step(tmp_path):
    """用极小网格跑通 flow -> particle -> checkpoint 的完整单步流程。

    网格足够小（6x6x6，无颗粒），单步耗时在毫秒级，用来在 CI 中低成本地
    验证核心模拟链路没有被破坏，而不是像此前那样完全跳过 server/核心路径。
    """
    from funlbm.lbm import create_lbm

    cache_dir = tmp_path / "data"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dx": 1.0,
                "dt": 1.0,
                "max_step": 1,
                "device": "cpu",
                "file": {
                    "cache_dir": str(cache_dir),
                    "custom": {"per_step": 1},
                    "checkpoint": {"per_step": 1},
                },
                "flow": {"size": [6, 6, 6], "param_type": "D3Q19"},
                "particles": [],
            }
        )
    )

    lbm = create_lbm(str(config_path))
    lbm.run(max_steps=1)

    assert lbm.step == 1
    assert (cache_dir / "checkpoint" / "checkpoint-0000000001.h5").exists()
    assert (cache_dir / "custom" / "custom-0000000001.h5").exists()
    assert (cache_dir / "constant.h5").exists()
