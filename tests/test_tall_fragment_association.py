import numpy as np
import pytest
from floorsight.contracts import TrackClass
from floorsight.vision.tracker import Tracker, _Track, Cluster


@pytest.mark.skip(reason='Preserved rejected height-only candidate: full GCP replay lost three true near-misses; requires richer geometry repair')
def test_narrow_tall_fragment_cannot_pull_established_person():
    tracker=Tracker(lambda x,y:1,(20.,20.))
    tr=_Track(1,'mover',np.array([5.,5.]),.1,0.)
    tr.confirmed=True
    tr.votes[TrackClass.person]=20
    fragment=Cluster(np.array([5.1,5.]),.14,.31,0.,['cam3'],height_m=2.53)
    person=Cluster(np.array([5.4,5.]),.14,.45,0.,['cam1'],height_m=1.9)
    assert tracker._assign([tr],[fragment],lambda tr:1.) == []
    assert tracker._assign([tr],[fragment,person],lambda tr:1.) == [(0,1)]
