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


def test_official_seven_dimensional_action():
    a=delta_pose_action([0,0,.1],[.05,-.02,.14],(7,))
    b=delta_pose_action([0,0,.1],[.05,-.02,.14],(1,7),gain=.75)
    assert a.shape==(7,) and b.shape==(1,7)
    assert a[-1]==-1 and np.max(np.abs(a))<=1
    assert not np.allclose(a[:3],b.reshape(-1)[:3])
    with pytest.raises(ValueError):
        delta_pose_action([0,0,0],[0,0,0],(4,))
