import inspect
import numpy as np
import pytest
from agcofl.teacher import cartesian_action
from agcofl.learning import Demonstrations

def test_no_mplib_planner():
    import agcofl.teacher as t
    src=inspect.getsource(t)
    assert 'from mani_skill.examples.motionplanning' not in src
    assert 'import mplib' not in src

def test_ee_action_shape():
    a=cartesian_action([0,0,.1],[.05,-.02,.15])
    assert a.shape==(4,)
    assert np.allclose(a,[.375,-.15,.375,-1.],atol=1e-6)
    assert cartesian_action([0,0,.1],[.05,-.02,.15],(1,4)).shape==(1,4)
    with pytest.raises(ValueError):
        cartesian_action([0,0,0],[0,0,0],(9,))

def test_reject_legacy_data(tmp_path):
    np.savez_compressed(tmp_path/'old.npz',success=True,
        state=np.zeros((1,27)),action=np.zeros((1,9)),flow=np.zeros((1,4,32,32)))
    with pytest.raises(RuntimeError,match='Legacy/incompatible'):
        Demonstrations(tmp_path)

def test_accept_new_data(tmp_path):
    np.savez_compressed(tmp_path/'new.npz',success=True,controller='pd_ee_delta_pos',
        state=np.zeros((1,27)),action=np.zeros((1,4)),flow=np.zeros((1,4,32,32)))
    ds=Demonstrations(tmp_path)
    assert len(ds)==1 and ds.action_dim==4
