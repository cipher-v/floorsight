import json
from scripts.benchmark.perception_compare import compare_jsonl


def test_equivalence_rejects_missing_frames_and_invalid_numbers(tmp_path):
    a, b = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    a.write_text(json.dumps({'gap': .5}) + '\n')
    b.write_text(json.dumps({'gap': .50001}) + '\n')
    assert compare_jsonl(a, b)['equal_within_tolerance']
    b.write_text('')
    assert not compare_jsonl(a, b)['equal_within_tolerance']
    b.write_text(json.dumps({'gap': float('nan')}) + '\n')
    assert not compare_jsonl(a, b)['equal_within_tolerance']
