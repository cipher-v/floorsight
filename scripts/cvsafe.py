"""Re-export of floorsight/cvsafe.py for the stand-alone perception scripts (loaded by file path, no package install)."""

import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("floorsight_cvsafe", _Path(__file__).resolve().parents[1] / "floorsight" / "cvsafe.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
clip_polygon, fill_poly, fill_convex = _m.clip_polygon, _m.fill_poly, _m.fill_convex
