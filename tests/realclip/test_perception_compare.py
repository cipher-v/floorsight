import json
from scripts.benchmark.perception_compare import compare_jsonl, environment_differences


def test_equivalence_rejects_missing_frames_and_invalid_numbers(tmp_path):
    a, b = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    a.write_text(json.dumps({'gap': .5}) + '\n')
    b.write_text(json.dumps({'gap': .50001}) + '\n')
    assert compare_jsonl(a, b)['equal_within_tolerance']
    b.write_text('')
    assert not compare_jsonl(a, b)['equal_within_tolerance']
    b.write_text(json.dumps({'gap': float('nan')}) + '\n')
    assert not compare_jsonl(a, b)['equal_within_tolerance']


def test_build_comparison_rejects_unrelated_dependency_changes():
    env = dict(python='3.12', machine='aarch64', numpy='2.5', scipy='1.18', torch='2.14', transformers='4.57', opencv='5.1')
    assert environment_differences(env, env | {'opencv': '5.0'}) == {}
    assert environment_differences(env, env | {'scipy': '1.15'}) == {
        'scipy': {'reference': '1.18', 'candidate': '1.15'}}
    assert environment_differences(env, {})
