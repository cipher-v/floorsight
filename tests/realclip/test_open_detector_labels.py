"""Open-vocabulary labels come back as phrases built from prompt tokens; they must still map to the right kind."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("open_detectors", Path(__file__).resolve().parents[2] / "scripts" / "open_detectors.py")
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)


def test_phrases_map_to_kinds():
    assert od.label_kind("forklift") == "forklift"
    assert od.label_kind("forklift truck") == "forklift"
    assert od.label_kind("forklift industrial vehicle") == "forklift"
    assert od.label_kind("industrial vehicle") == "forklift"
    assert od.label_kind("stack crates") == "load"
    assert od.label_kind("box pallet") == "load"
    assert od.label_kind("person") == "person"
    assert od.label_kind("") is None and od.label_kind("truck") is None
