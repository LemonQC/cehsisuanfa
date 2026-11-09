import numpy as np
import torch
from agcofl.representation import future_masks,mask_route_side,route_points
from agcofl.partial_supervision import torch_segment_loss


def test_ref_routes_encode_different_sides():
    for s in (-1,1):
        f=future_masks(np.array([-.145,0.]),s)
        assert f.shape==(4,32,32)
        assert mask_route_side(f)==s
        assert set(np.unique(f)).issubset({0,1})
        assert np.allclose(route_points(s)[0],[-.145,0.])


def test_unselected_frames_have_zero_direct_gradient():
    logits=torch.zeros((8,32,32),requires_grad=True)
    labels=torch.zeros_like(logits);labels[:,12:17,12:17]=1
    loss=torch_segment_loss(logits,labels,'end');loss.backward()
    grad=logits.grad.abs().sum(dim=(1,2))
    assert torch.all(grad[:-1]==0)
    assert grad[-1]>0
    logits.grad.zero_()
    loss=torch_segment_loss(logits,labels,'anchor');loss.backward()
    grad=logits.grad.abs().sum(dim=(1,2))
    assert int((grad>0).sum())<=2
    assert grad[-1]>0


def test_loss_falls_with_correct_prediction():
    labels=torch.zeros((6,16,16))
    labels[:,4:8,5:9]=1
    bad=torch.zeros_like(labels)
    good=torch.where(labels>0,torch.tensor(7.),torch.tensor(-7.))
    for mode in ['full','anchor','end']:
        assert torch_segment_loss(good,labels,mode) < torch_segment_loss(bad,labels,mode)
