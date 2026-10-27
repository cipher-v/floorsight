"""Compare actual rounded privacy-region coverage after numeric benchmark differences.

This checks covered pixels, not anonymity or blur strength. No images are read or
written. It does not change the strict metadata comparison in the benchmark.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def mask(row, size):
    w, h = size
    covered = np.zeros((h, w), bool)
    for region in row['heads'] + row['faces']:
        x1,y1,x2,y2 = (int(round(v)) for v in region)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if x2-x1 >= 2 and y2-y1 >= 2:
            covered[y1:y2,x1:x2] = True
    return covered


def compare(reference, candidate, size):
    before = [json.loads(x) for x in reference.read_text().splitlines()]
    after = [json.loads(x) for x in candidate.read_text().splitlines()]
    if len(before) != len(after) or [r['f'] for r in before] != [r['f'] for r in after]:
        return {'same_frames': False, 'identical_covered_pixels': False}
    lost = gained = 0
    for a, b in zip(before, after):
        ma, mb = mask(a, size), mask(b, size)
        lost += int(np.count_nonzero(ma & ~mb))
        gained += int(np.count_nonzero(mb & ~ma))
    return {'same_frames': True, 'frames': len(before), 'reference_pixels_uncovered': lost,
            'additional_covered_pixels': gained, 'identical_covered_pixels': lost == gained == 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    checks = []
    for c in report['comparisons']:
        base = a.report.parent / f"{c['clip']}-{c['repeat']}-{c['reference']}"
        candidate = a.report.parent / f"{c['clip']}-{c['repeat']}-{c['label']}"
        result = json.loads((base/f"{c['clip']}-result.json").read_text())
        name = f"{c['clip']}-privacy-regions.jsonl"
        checks.append({k:c[k] for k in ['label','reference','clip','repeat']} |
                      compare(base/name, candidate/name, result['frameSize']))
    a.out.write_text(json.dumps({'schema':'floorsight.privacy-pixel-comparison.v1', 'comparisons':checks},indent=2))
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
