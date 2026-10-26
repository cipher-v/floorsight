"""Evaluation-only pixel/map review. Truth labels never enter the product pipeline."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from floorsight.site import load_site
from floorsight.vision.calib import MarkerDetector, marker_floor_corners, fit_calibration, merge_observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--times', nargs='+', type=float, default=[55.9, 104., 149.3])
    p.add_argument('--review-name', default=None)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sim = root / 'data/sim/demo_001'
    out = root / 'eval/reviews' / (a.review_name or a.run)
    out.mkdir(parents=True, exist_ok=False)
    worlds = [json.loads(line)['world'] for line in (root / 'data/runtime' / a.run / 'worlds.jsonl').read_text().splitlines()]
    truth = [json.loads(line) for line in (sim / 'truth_frames.jsonl').read_text().splitlines()]
    site = load_site(str(sim / 'site.json'))
    detector = MarkerDetector(site)
    corners = marker_floor_corners(site)
    manifest = {'run': a.run, 'representation': '2D source video overlays and metric floor map', 'provenance': 'Locally generated simulator pixels; marker-fitted projection. Green crosses are evaluation truth only.', 'frames': []}
    for target in a.times:
        world = min(worlds, key=lambda w: abs(w['t'] - target))
        gt = min(truth, key=lambda w: abs(w['t'] - world['t']))
        tiles = []
        for info in site.cameras:
            cap = cv2.VideoCapture(str(sim / f'{info.camera_id}.mp4'))
            calib = None
            calibrated_t = None
            # Last successful measured marker pose, never the simulator camera oracle.
            for sample_t in np.arange(0., world['t'], 2.):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(sample_t * info.fps))
                valid, frame = cap.read()
                if valid:
                    fitted = fit_calibration(detector.detect(frame, fine=True), corners, info)
                    if fitted is not None:
                        calib, calibrated_t = fitted, float(sample_t)
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(world['t'] * info.fps))
            ok, img = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f'Missing frame {info.camera_id} {world["t"]}')
            fitted = fit_calibration(detector.detect(img, fine=True), corners, info)
            if fitted is not None:
                calib, calibrated_t = fitted, world['t']
            if calib is None:
                raise RuntimeError(f'Marker calibration failed at {world["t"]} {info.camera_id}')
            for tr in world['tracks']:
                uv, z = calib.project(np.array([[tr['x'], tr['y'], 0.]]))
                u, v = np.round(uv[0]).astype(int)
                if z[0] > 0 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    col = (0, 180, 255) if tr['state'] != 'seen' else (255, 150, 0)
                    cv2.circle(img, (u, v), 9, col, 2)
                    cv2.putText(img, f'{tr["track_id"]} {tr["state"]}', (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1)
            for ent in gt['entities']:
                uv, z = calib.project(np.array([[ent['x'], ent['y'], 0.]]))
                u, v = np.round(uv[0]).astype(int)
                if z[0] > 0 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.drawMarker(img, (u, v), (50, 255, 50), cv2.MARKER_CROSS, 18, 2)
                    cv2.putText(img, ent['id'], (u+8, v+15), cv2.FONT_HERSHEY_SIMPLEX, .45, (50, 255, 50), 1)
            cv2.rectangle(img, (0, 0), (img.shape[1], 32), (22, 26, 30), -1)
            cv2.putText(img, f'{info.camera_id} t={world["t"]:.1f}s pose@{calibrated_t:.1f} | circles=product; green=TRUTH', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
            tiles.append(cv2.resize(img, (768, 432)))
        panel = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        path = out / f't{world["t"]:.1f}.jpg'
        cv2.imwrite(str(path), panel)
        manifest['frames'].append({'t': world['t'], 'preview': str(path), 'source': str(sim), 'tracks': world['tracks'], 'truth': gt['entities'], 'defects': 'Review required; projection through occluders is intentionally shown, not a visibility claim.'})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    html = '<!doctype html><meta charset="utf-8"><title>Floorsight evidence review</title><style>body{background:#111923;color:#edf3ff;font:16px system-ui;margin:32px}img{max-width:100%}p{max-width:85ch}</style><h1>Floorsight source-to-map review</h1><p>Simulated footage. Blue circles are reported observed floor positions; amber circles are uncertain positions. Green crosses are independent evaluation truth. These overlays project points through occluders and do not assert visibility. Marker calibration is fitted from each displayed frame. This is 2D evidence, not a 3D reconstruction.</p>'
    for frame in manifest['frames']:
        html += f'<h2>{frame["t"]:.1f} seconds</h2><img src="{Path(frame["preview"]).name}">'
    (out / 'index.html').write_text(html, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
