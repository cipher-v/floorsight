"""Verify exact calibration backgrounds on private clips without saving pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.real_clip_perception import background_samples, temporal_median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for clip in a.clip:
        cap = cv2.VideoCapture(str(clip))
        samples = background_samples(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), mode='seek')
        cap.release()
        for repeat, order in enumerate([['numpy', 'compact'], ['compact', 'numpy']]):
            measured = {}
            for mode in order:
                start = time.perf_counter()
                background = temporal_median(samples, mode=mode)
                elapsed = time.perf_counter() - start
                measured[mode] = {'seconds': elapsed, 'sha256': hashlib.sha256(background.tobytes()).hexdigest()}
            rows.append({'clip': clip.stem, 'samples': len(samples), 'repeat': repeat, 'runs': measured,
                         'identical': measured['numpy']['sha256'] == measured['compact']['sha256']})
        del samples
    report = {'schema': 'floorsight.median-bench.v1', 'opencv': cv2.__version__, 'runs': rows}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all(r['identical'] and r['samples'] == 25 for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
