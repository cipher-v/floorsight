"""Run the complete real-clip pipeline with interleaved build/scheduling variants.

No detector cache is replayed. Decode, all detector models, privacy filtering,
floor processing and video encoding are included. Preserve both fast AND slow
runs. Compare outputs before claiming a speedup. Inputs stay on the host.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def compare_jsonl(a, b, tolerance=.001):
    left = [json.loads(s) for s in a.read_text().splitlines()]
    right = [json.loads(s) for s in b.read_text().splitlines()]
    changed, maximum = [], 0.
    def visit(x, y, path):
        nonlocal maximum
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            delta = abs(x-y)
            maximum = max(maximum, delta)
            if not math.isfinite(delta) or delta > tolerance:
                changed.append(path)
        elif isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for k in x:
                visit(x[k], y[k], path + '/' + k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, path + '/' + str(i))
        elif x != y:
            changed.append(path)
    visit(left, right, '')
    return {'equal_within_tolerance': not changed, 'tolerance': tolerance,
            'max_numeric_delta': maximum, 'changed_fields': len(changed), 'examples': changed[:12],
            'reference_sha256': digest(a), 'candidate_sha256': digest(b)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', action='append', required=True, help='label,python-wrapper,workers,threads[,torch-layout,background-sampling,background-median]')
    p.add_argument('--clip', type=Path, action='append', required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--hf-index', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vocab-every', type=int, default=5)
    args = p.parse_args()
    variants = [v.split(',') for v in args.variant]
    if len(variants) < 2 or any(len(v) not in (4, 5, 6, 7) for v in variants) or args.frames < 1 or args.repeats < 1:
        p.error('at least two variants need four to seven fields; frames and repeats must be positive')
    variants = [v + ['contiguous', 'seek', 'compact'][len(v)-4:] for v in variants]
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'floorsight.perception-compare.v1',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'machine': platform.machine(),
        'protocol': 'Same input frames/models/thresholds. Full processes run sequentially. Variant order reversed each repeat.',
        'repeats': args.repeats, 'frames_per_clip': args.frames,
        'model_index': json.loads(args.hf_index.read_text()),
        'calibration_sha256': digest(args.calibration),
        'source_sha256': {str(q.relative_to(ROOT)): digest(q) for q in
            [ROOT/'scripts/open_detectors.py', ROOT/'scripts/real_clip_perception.py', Path(__file__)]},
        'runs': [], 'comparisons': []}
    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for repeat in range(args.repeats):
        for clip in args.clip:
            reference = None
            run_outputs = []
            for label, py, workers, threads, layout, background, median in variants[::1 if repeat % 2 == 0 else -1]:
                output = args.out / f'{clip.stem}-{repeat}-{label}'
                command = [py, str(ROOT/'scripts/real_clip_perception.py'), '--clip', str(clip),
                    '--name', clip.stem, '--output', str(output), '--models', str(args.models),
                    '--hf-index', str(args.hf_index), '--calibration', str(args.calibration),
                    '--max-frames', str(args.frames), '--vocab', 'gdino', '--segmenter', 'slimsam77',
                    '--person-model', 'rtdetrv2', '--privacy-extra', 'yolox_s', '--vehicle-source', 'confirmed',
                    '--vocab-every', str(args.vocab_every), '--detector-workers', workers, '--cpu-threads', threads,
                    '--torch-layout', layout, '--background-sampling', background, '--background-median', median]
                start = time.perf_counter()
                with (args.out/f'{output.name}.log').open('w') as log:
                    try:
                        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                        exit_code = proc.returncode
                    except (OSError, subprocess.TimeoutExpired) as error:
                        log.write(f'\n{type(error).__name__}: {error}\n')
                        exit_code = -1
                row = {'label': label, 'clip': clip.stem, 'clip_sha256': digest(clip), 'repeat': repeat,
                       'command': command, 'exit_code': exit_code, 'process_wall_s': time.perf_counter()-start}
                if exit_code == 0:
                    result = json.loads((output/f'{clip.stem}-result.json').read_text())
                    row.update(result_seconds=result['seconds'], timing=result['timing'], environment=result['environment'],
                        privacy=result['privacy'], encoder_exit=result['encoderExit'])
                    run_outputs.append((label, output))
                    if label == variants[0][0]:
                        reference = output
                report['runs'].append(row)
                save()
                print(json.dumps({k: row[k] for k in ['label','clip','repeat','exit_code','process_wall_s']}), flush=True)
            if reference:
                for label, output in run_outputs:
                    if output == reference:
                        continue
                    comparison = {'label': label, 'reference': variants[0][0], 'clip': clip.stem, 'repeat': repeat}
                    for kind in ['detections', 'privacy-regions']:
                        name = f'{clip.stem}-{kind}.jsonl'
                        comparison[kind] = compare_jsonl(reference/name, output/name)
                    report['comparisons'].append(comparison)
            save()
    report['median_process_wall_s'] = {v[0]: statistics.median([r['process_wall_s'] for r in report['runs']
        if r['label'] == v[0] and r['exit_code'] == 0]) for v in variants
        if any(r['label'] == v[0] and r['exit_code'] == 0 for r in report['runs'])}
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['outputs_equivalent'] = all(c[k]['equal_within_tolerance'] for c in report['comparisons']
                                       for k in ['detections', 'privacy-regions'])
    report['all_frames_processed'] = all(r.get('timing', {}).get('detector_stages_s', {}).get('frames') == args.frames
                                          for r in report['runs'])
    report['encoders_succeeded'] = all(r.get('encoder_exit') and all(v == 0 for v in r['encoder_exit'].values())
                                       for r in report['runs'])
    save()
    if not report['outputs_equivalent'] or not report['all_frames_processed'] or not report['encoders_succeeded'] or any(
            r['exit_code'] != 0 or r.get('privacy', {}).get('framesWithheld', 1) != 0 for r in report['runs']):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
