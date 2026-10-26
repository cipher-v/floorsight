"""Public test fixtures: no private data, cloud account or setup run required."""
import pytest
import cv2

from floorsight.contracts import SitePlan
from floorsight.sim.scenario import demo_cameras, demo_site

# Camera workers already run concurrently, as in the application CLI. Avoid
# multiplying each worker by every host CPU during detector integration tests.
cv2.setNumThreads(1)


@pytest.fixture
def demo_plan():
    return SitePlan.model_validate(demo_site(demo_cameras()))
