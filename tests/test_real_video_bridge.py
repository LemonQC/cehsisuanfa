"""No simulator required: validate real MP4 parsing and RoboReel action interface."""
import cv2
import numpy as np
import pytest
from agcofl.real_video_flow import blue_candidate, extract_video_flow
from agcofl.roboreel_bridge import delta_pose_action


def test_blue_candidate_present_absent():
    rgb=np.zeros((128,128,3),np.uint8)
    mask,pos=blue_candidate(rgb)
    assert pos is None and not mask.any()
    rgb[45:75,54:80,:]=[10,35,210]
    mask,pos=blue_candidate(rgb)
    assert pos is not None and int(mask.sum())>200
    assert .4<pos[0]<.6 and .4<pos[1]<.6


def test_mp4_masks_and_provenance(tmp_path):
    p=tmp_path/'video.mp4'
    w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),15,(128,128))
    if not w.isOpened():
        pytest.skip('No mp4v encoder')
    for i in range(30):
        rgb=np.zeros((128,128,3),np.uint8)
        x=22+i
        rgb[48:70,x:x+22,:]=[15,40,225]
        w.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    w.release()
    flow,meta=extract_video_flow(p,tmp_path/'out')
    assert flow.shape==(8,32,32)
    assert meta['accepted'] and meta['coverage']>.8
    assert meta['normalized_motion_path']>0
    assert (tmp_path/'out'/'human_video_diagnostic.json').exists()
    assert (tmp_path/'out'/'mask_contact_sheet.png').exists()
    assert (tmp_path/'out'/'human_mask_overlay.png').exists()


def test_official_seven_dimensional_action():
    a=delta_pose_action([0,0,.1],[.05,-.02,.14],(7,))
    b=delta_pose_action([0,0,.1],[.05,-.02,.14],(1,7),gain=.75)
    assert a.shape==(7,) and b.shape==(1,7)
    assert a[-1]==-1 and np.max(np.abs(a))<=1
    assert not np.allclose(a[:3],b.reshape(-1)[:3])
    with pytest.raises(ValueError):
        delta_pose_action([0,0,0],[0,0,0],(4,))


def test_push_phase_measures_planar_cube_motion_without_shape_error():
    """Regression: previous push frame subtracted 2D cube XY from 3D prior pose."""
    from types import SimpleNamespace
    from agcofl.roboreel_bridge import DiagnosticPushTeacher

    cube = np.array([0.01, -0.02, 0.02], dtype=np.float32)
    obj = SimpleNamespace(pose=SimpleNamespace(p=cube))
    agent = SimpleNamespace(tcp_pose=SimpleNamespace(p=np.array([-0.08, -0.02, 0.028])))
    base = SimpleNamespace(
        obj=obj, goal_pos=np.array([0.15, 0., 0.]),
        agent=agent, cube_half_size=.02
    )
    env = SimpleNamespace(unwrapped=base, action_space=SimpleNamespace(shape=(7,)))
    teacher = DiagnosticPushTeacher(env)
    teacher.phase = 'push'
    teacher.phase_steps = 24
    teacher.last_cube = np.array([0., -0.02, 0.02], dtype=np.float32)
    a1 = teacher.act()
    assert a1.shape == (7,)
    assert np.isfinite(a1).all()
    assert teacher.stale_steps == 0
    assert teacher.last_cube.shape == (3,)

    # Cube stays still in the next sample: should accumulate one stall step.
    a2 = teacher.act()
    assert a2.shape == (7,)
    assert teacher.stale_steps == 1


def test_push_no_progress_still_transitions_to_recover():
    from types import SimpleNamespace
    from agcofl.roboreel_bridge import DiagnosticPushTeacher

    xyz = np.array([.02, 0., .02], dtype=np.float32)
    agent = SimpleNamespace(tcp_pose=SimpleNamespace(p=np.array([-.08, 0., .028])))
    base = SimpleNamespace(obj=SimpleNamespace(pose=SimpleNamespace(p=xyz)),
                           goal_pos=np.array([.15, 0., 0.]),
                           agent=agent, cube_half_size=.02)
    env = SimpleNamespace(unwrapped=base, action_space=SimpleNamespace(shape=(7,)))
    teacher = DiagnosticPushTeacher(env)
    teacher.phase = 'push'
    teacher.phase_steps = 24
    teacher.last_cube = xyz.copy()
    teacher.stale_steps = 17
    action = teacher.act()
    assert action.shape == (7,)
    assert teacher.phase == 'recover'
    assert teacher.attempt == 1
