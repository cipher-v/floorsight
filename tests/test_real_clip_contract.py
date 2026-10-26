"""Admission boundary tests; execute on coordinated GCP only."""
from copy import deepcopy
import pytest
from floorsight.real_clip_contract import assess


def ref(path):
    return dict(path=path, sha256="a"*64)  # Deliberately synthetic fixture, never a receipt.


def reviewed():
    return dict(schema="floorsight.real-clip.v1", camera_id="camera9",
                source=dict(url="https://data.mendeley.com/datasets/xjmtb22pff/1",
                            license="CC-BY-4.0", attribution="Onal and Dandil"),
                original=ref("quarantine/source.mp4"),
                privacy=dict(status="reviewed",filtered=ref("filtered/clip.webm"),
                             review_evidence=ref("evidence/privacy.json"),fail_closed_check=True))


def test_reviewed_uncalibrated_clip_allows_pixels_but_not_metric_events():
    result=assess(reviewed())
    assert result.preview and result.pixel_observations and not result.metric_events


def test_zero_detected_faces_is_not_privacy_approval():
    m=reviewed();m['privacy']={'faces_detected':0,'status':'processed'}
    assert not assess(m).preview


@pytest.mark.parametrize('path',['../private.mp4','/raw.mp4','C:/raw.mp4','media\\raw.mp4','',
                                  '.', './clip.webm', 'media//clip.webm', 'media/clip\n.webm'])
def test_invalid_dependency_path_is_rejected(path):
    m=reviewed();m['privacy']['filtered']['path']=path
    assert not assess(m).preview


def test_raw_file_cannot_be_relabelled_filtered():
    m=reviewed();m['privacy']['filtered']=deepcopy(m['original'])
    assert not assess(m).preview


def test_scale_and_heldout_geometry_are_required():
    m=reviewed();m['calibration']=dict(status='validated',units='metres',file=ref('calib/camera.json'),
        heldout_reprojection_p95_px=2.,acceptance_limit_px=3.,heldout_evidence=ref('eval/geometry.json'),
        scale_measurement=ref('calib/measurement.json'))
    assert assess(m).metric_events
    for key in ['scale_measurement','heldout_evidence']:
        broken=deepcopy(m);broken['calibration'].pop(key)
        assert not assess(broken).metric_events
    m['calibration']['heldout_reprojection_p95_px']=float('nan')
    assert not assess(m).metric_events
