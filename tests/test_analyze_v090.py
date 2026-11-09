"""No network and no ManiSkill required."""
import numpy as np
from agcofl.analyze_v090 import move_mask,generate,mask_iou,mask_center


def test_constant_velocity_exact_linear_small_square():
    masks=np.zeros((2,32,32),np.float32)
    masks[0,10:14,10:14]=1
    masks[1,10:14,12:16]=1
    cv=generate(masks,"constant_velocity")
    for t in range(8):
        expected=move_mask(masks[1],2*(t+1),0)
        assert np.array_equal(cv[t],expected)
    assert mask_iou(cv[0],move_mask(masks[1],2,0))==1.0
    assert mask_iou(masks[1],cv[0])<1.


def test_persistence_keeps_full_shape_and_geometry():
    past=np.zeros((2,32,32),np.float32)
    past[0,7:11,5:9]=1
    past[1,7:11,6:10]=1
    pred=generate(past,"persistence")
    assert pred.shape==(8,32,32)
    assert np.array_equal(pred[7],past[1])
    assert pred.sum()==8*past[1].sum()


def test_shift_has_zero_padding_not_toroidal_wrap():
    mask=np.zeros((32,32),np.float32)
    mask[15,31]=1
    out=move_mask(mask,2,0)
    assert out.sum()==0.
    assert mask_center(out) is None
    assert mask_iou(np.zeros_like(mask),np.zeros_like(mask))==1.
