"""Fixtures for the server tests: small simulator runs generated on the fly."""

from __future__ import annotations

import contextlib
import io
import os

import pytest

from floorsight.sim.scenario import run as run_scenario


def _quiet_run(out_dir: str, duration_s: float, write_video: bool) -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        run_scenario(out_dir, duration_s=duration_s, write_video=write_video)
    return out_dir


@pytest.fixture(scope="session")
def sim_truth_dir(tmp_path_factory) -> str:
    """6 s of simulator ground truth, no video (for --source fake)."""
    return _quiet_run(str(tmp_path_factory.mktemp("sim_truth")), 6.0, write_video=False)


@pytest.fixture(scope="session")
def sim_video_dir(tmp_path_factory) -> str:
    """1 s of rendered camera video plus ground truth (for --source sim)."""
    out = _quiet_run(str(tmp_path_factory.mktemp("sim_video")), 1.0, write_video=True)
    assert os.path.exists(os.path.join(out, "cam1.mp4"))
    return out
