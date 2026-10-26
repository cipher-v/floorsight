import os
import sys

import pytest

# Make the repo root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="session")
def sim_dir(tmp_path_factory):
    """A fresh 180 s scripted shift (truth files only, no video) - takes ~2 s."""
    from floorsight.sim.scenario import run

    out = tmp_path_factory.mktemp("sim")
    run(str(out), 180.0, write_video=False)
    return str(out)


@pytest.fixture(scope="session")
def site(sim_dir):
    from floorsight.site import load_site

    return load_site(os.path.join(sim_dir, "site.json"))


@pytest.fixture(scope="session")
def full_run(sim_dir):
    """The agent driven over the whole shift with every approval auto-approved."""
    from floorsight.agent.demo import run

    os.environ.pop("FLOORSIGHT_LLM", None)
    return run(sim_dir, auto_approve=True)
