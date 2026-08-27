"""Lightweight smoke tests for funlbm.

funlbm is a Lattice-Boltzmann-Method (LBM) numerical simulation package built
on numpy/scipy/torch/h5py. These tests intentionally avoid running any real
simulation (heavy compute) and avoid touching real data files or GUI/plot
output; they only check that the package installs correctly and that the
handful of submodules which *can* import cleanly behave sanely on trivial
inputs.

Known upstream bugs discovered while writing this suite (NOT fixed here,
per task scope -- only dependency-declaration gaps were fixed in
pyproject.toml):

1. ``funlbm.server.submit`` / ``funlbm.server.update`` (and therefore
   ``funlbm.server`` and the ``funlbm`` CLI entry point) do
   ``from funbuild.shell import run_shell``. The currently published
   ``funbuild`` (1.6.69) has no ``shell`` submodule at all -- its API has
   drifted. This also requires a source change, out of scope here.
2. ``funlbm.config.base.BoundaryConfig.__init__`` does
   ``Boundary(**input)`` / ``Boundary(**output)`` / ``Boundary(**back)``
   without an ``or {}`` fallback (unlike ``forward``/``bottom``/``top``
   a few lines below, which do have the fallback). Calling
   ``BoundaryConfig()`` with all-default arguments raises ``TypeError``.
"""

import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Basic package import
# ---------------------------------------------------------------------------


def test_import_top_level_package():
    import funlbm  # noqa: F401


def test_import_funlbm_config():
    import funlbm.config  # noqa: F401
    import funlbm.config.base  # noqa: F401


def test_import_funlbm_file():
    import funlbm.file  # noqa: F401


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
    """BoundaryConfig works when input/output/back are passed explicitly
    (the only way to avoid the default-args bug, see module docstring)."""
    from funlbm.config.base import BoundaryConfig

    cfg = BoundaryConfig(input={}, output={}, back={})
    for face in ("input", "output", "back", "forward", "bottom", "top"):
        boundary = getattr(cfg, face)
        assert boundary.condition.name == "WALL"


def test_boundary_config_default_construction_bug():
    """Documents a real bug: BoundaryConfig() with pure defaults crashes
    because `input`/`output`/`back` are unpacked with `**` without an
    `or {}` fallback (see module docstring, item 3). Not fixed here."""
    from funlbm.config.base import BoundaryConfig

    with pytest.raises(TypeError):
        BoundaryConfig()


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
    import json

    from funlbm.config.base import BaseConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"foo": "bar"}))

    cfg = BaseConfig().from_file(str(config_path))
    assert cfg.get("foo") == "bar"


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
# Submodules previously blocked by the funlog/farlog naming collision
# (issue #153, fixed: `from funlog import getLogger` -> `from farlog import
# getLogger`). These now import cleanly.
# ---------------------------------------------------------------------------

_FUNBUILD_SHELL_BUG_REASON = (
    "无法导入：源码中 `from funbuild.shell import run_shell` "
    "（server/submit.py, server/update.py）在当前已发布的 funbuild 1.6.69 中"
    "不存在 shell 子模块，属于上游 API 漂移。这是源码 bug，非依赖声明问题，"
    "本次任务范围内未修复，仅记录。"
)


@pytest.mark.parametrize(
    "module_name",
    [
        "funlbm.util",
        "funlbm.base",
        "funlbm.flow",
        "funlbm.particle",
        "funlbm.lbm",
    ],
)
def test_modules_import_cleanly(module_name):
    import importlib

    importlib.import_module(module_name)


def test_server_module_blocked_by_funbuild_api_drift():
    pytest.importorskip("funlbm.server", reason=_FUNBUILD_SHELL_BUG_REASON)
    pytest.fail(
        "funlbm.server imported successfully -- the funbuild.shell bug "
        "documented in this test's skip reason appears to be fixed "
        "upstream; please replace this skip with a real smoke test."
    )


def test_cli_entry_point_help():
    """The `funlbm` console-script entry point (funlbm.server:funlbm)
    currently cannot even start because funlbm.server fails to import
    (see test_server_module_blocked_by_funbuild_api_drift). We invoke it
    via `python -m funlbm.server` equivalent (the installed console
    script) and skip with a clear reason instead of faking a pass."""
    result = subprocess.run(
        [sys.executable, "-c", "from funlbm.server import funlbm"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "funlbm CLI entry point cannot be imported: "
            + _FUNBUILD_SHELL_BUG_REASON
        )
    pytest.fail(
        "funlbm CLI entry point imported successfully -- please replace "
        "this skip with a real `--help` subprocess test."
    )
