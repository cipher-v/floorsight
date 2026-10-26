"""The site plan's marker dictionary is used, and an unknown one is refused (audit R5). Synthetic images only."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from floorsight.contracts import FloorMarker, SitePlan
from floorsight.edge.markers import get_dictionary
from floorsight.vision.calib import MarkerDetector, site_dictionary


def _site(*dictionaries: str) -> SitePlan:
    markers = [FloorMarker(marker_id=i, dictionary=d, center=(1.0 + i, 1.0), size_m=0.6) for i, d in enumerate(dictionaries)]
    return SitePlan(site_id="t", name="t", width_m=10.0, depth_m=10.0, zones=[], obstacles=[], markers=markers, cameras=[])


def _image(dict_name: str, marker_id: int) -> np.ndarray:
    tag = cv2.aruco.generateImageMarker(get_dictionary(dict_name), marker_id, 240, borderBits=1)
    img = np.full((480, 640), 255, np.uint8)
    img[120:360, 200:440] = tag
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_default_dictionary_is_unchanged():
    assert FloorMarker(marker_id=0, center=(0.0, 0.0), size_m=0.6).dictionary == "DICT_4X4_50"
    assert site_dictionary(_site()) == "DICT_4X4_50"  # no markers: the default
    found = MarkerDetector(_site("DICT_4X4_50", "DICT_4X4_50")).detect(_image("DICT_4X4_50", 1))
    assert list(found) == [1]


def test_configured_dictionary_is_used():
    site = _site("DICT_5X5_100", "DICT_5X5_100")
    assert site_dictionary(site) == "DICT_5X5_100"
    assert list(MarkerDetector(site).detect(_image("DICT_5X5_100", 1))) == [1]
    # The hard-coded 4x4 detector this replaces finds nothing in the same frame.
    assert MarkerDetector(_site("DICT_4X4_50", "DICT_4X4_50")).detect(_image("DICT_5X5_100", 1)) == {}


@pytest.mark.parametrize("name", ["DICT_4X4_51", "dict_4x4_50", "ArucoDetector", "", None])
def test_unknown_dictionary_name_is_refused(name):
    with pytest.raises(ValueError, match="unknown ArUco dictionary"):
        get_dictionary(name)


def test_site_with_unknown_dictionary_is_refused():
    with pytest.raises(ValueError, match="unknown ArUco dictionary 'DICT_9X9_7'"):
        MarkerDetector(_site("DICT_9X9_7"))


def test_mixed_dictionaries_are_refused():
    with pytest.raises(ValueError, match="mix ArUco dictionaries"):
        MarkerDetector(_site("DICT_4X4_50", "DICT_5X5_100"))
