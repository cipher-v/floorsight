import json
from scripts.benchmark.privacy_coverage_compare import compare


def test_coverage_compares_rounded_pixels_and_detects_lost_coverage(tmp_path):
    a,b=tmp_path/'a.jsonl',tmp_path/'b.jsonl'
    a.write_text(json.dumps({'f':0,'heads':[[1.1,1,4,4]],'faces':[]})+'\n')
    b.write_text(json.dumps({'f':0,'heads':[[1.2,1,4,4]],'faces':[]})+'\n')
    assert compare(a,b,(8,8))['identical_covered_pixels']
    b.write_text(json.dumps({'f':0,'heads':[[2,1,4,4]],'faces':[]})+'\n')
    assert compare(a,b,(8,8))['reference_pixels_uncovered'] == 3
