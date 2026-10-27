import json
from types import SimpleNamespace
import pytest
from floorsight.realclip import ReplayVision


@pytest.mark.parametrize('value,returncode', [(float('nan'), 0), (float('inf'), 0), (-1, 0), ('invalid', 0), (.5, 1)])
def test_invalid_or_failed_second_measurement_is_inconclusive(tmp_path, monkeypatch, value, returncode):
    monkeypatch.setenv('FLOORSIGHT_REDETECT_CMD', 'detector')
    monkeypatch.setattr('floorsight.realclip.subprocess.run', lambda *a, **kw:
        SimpleNamespace(stdout=json.dumps({'clearance_m': value}), stderr='', returncode=returncode))
    vision = ReplayVision.__new__(ReplayVision)
    temp = tmp_path/'request'; temp.mkdir()
    result = vision._redetect_run({'annotated': str(temp/'absent.png')}, str(temp), .55, 1.)
    assert result.supports is None
    assert not temp.exists()
