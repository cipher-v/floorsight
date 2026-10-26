"""Replay a real-clip replay folder's floor records through two versions of the forklift interlock and list what each
would have commanded (audit C1, 27 Oct: does the stricter resume rule change the recorded app run?).

    python scripts/interlock_replay_check.py REPLAY_DIR OLD_INTERLOCK_PY [--out result.json]

No video is decoded and no detector runs: the WorldStates are built from the saved per-frame floor records exactly as
the app's replay vision builds them, one per video frame, at the same timestamps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from floorsight.contracts import FrameMeta  # noqa: E402
from floorsight.interfaces import FramePacket
from floorsight.interlock import ForkliftInterlock
from floorsight.realclip import ReplayVision, load_playlist
from floorsight.site import load_site


def _old_interlock(path: str):
    spec = importlib.util.spec_from_file_location("interlock_old", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ForkliftInterlock()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay_dir")
    ap.add_argument("old_interlock")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rd = Path(a.replay_dir)
    pl = load_playlist(rd)
    vision = ReplayVision(load_site(str(rd / "site.json")), None, str(rd))
    locks = {"old": _old_interlock(a.old_interlock), "new": ForkliftInterlock()}
    cmds: dict[str, list] = {k: [] for k in locks}
    img = np.zeros((108, 192, 3), np.uint8)
    for clip in pl["clips"]:
        cap = cv2.VideoCapture(str(rd / clip["video"]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for local in range(n):
            meta = FrameMeta(camera_id=pl["camera_id"], frame_index=clip["offset_frames"] + local,
                             t=round(clip["offset_t"] + local / clip["fps"], 3), width=192, height=108, faces_blurred=0, blur_applied=True)
            world = vision.process({pl["camera_id"]: FramePacket(meta, img)})
            for k, lock in locks.items():
                for c in lock.update(world):
                    cmds[k].append({"clip": clip["name"], "t": c.get("t"), "command": c.get("command"), "vehicle": c.get("vehicle"),
                                    "person": c.get("person"), "reason": c.get("reason")})
    for k in locks:
        print(f"--- {k}: {len(cmds[k])} commands")
        for c in cmds[k]:
            print(f"  {c['clip']:<14} t={c['t']:>7} {c['command']:<7} {c['vehicle']} / {c['person']}: {c['reason']}")
    same = [(c["t"], c["command"]) for c in cmds["old"]] == [(c["t"], c["command"]) for c in cmds["new"]]
    print("SAME_COMMANDS" if same else "DIFFERENT_COMMANDS")
    if a.out:
        a.out.write_text(json.dumps({"same": same, **cmds}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
