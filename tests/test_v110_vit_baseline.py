"""Fast CPU regressions for the REAL pretrained-ViT architecture integration.

These synthetic numeric tensors are UNIT TEST inputs ONLY, never training data,
performance evidence or a replacement for downloaded pretrained weights.
"""
import numpy as np
import pytest
import torch
from agcofl.vit_flow_pilot import (
    TemporalViTFlow, baseline_predict, build_windows, select_loss, video_split
)


def test_pretrained_vit_head_shape_and_no_future_input():
    torch.manual_seed(42)
    net=TemporalViTFlow(input_dim=8,d_model=32,horizon=8)
    frozen_features=torch.randn(2,2,8)
    observed_centers=torch.tensor([[[.1,.2],[.11,.20]],
                                   [[.15,.3],[.16,.30]]],dtype=torch.float32)
    prediction=net(frozen_features,observed_centers)
    assert prediction.shape==(2,8,2)
    assert torch.allclose(prediction,observed_centers[:,-1:,:].expand(-1,8,-1))
    label=torch.zeros(2,8,2)
    loss=select_loss(prediction,label,"full")
    loss.backward()
    assert net.output.weight.grad is not None
    assert torch.isfinite(net.output.weight.grad).all()


def test_supervision_modes_have_different_gradients():
    p=torch.randn(2,8,2,requires_grad=True)
    truth=torch.zeros_like(p)
    grads={}
    for mode in ("full","anchor","end"):
        loss=select_loss(p,truth,mode)
        grads[mode]=torch.autograd.grad(loss,p,retain_graph=True)[0]
    assert torch.count_nonzero(grads["full"]).item()==32
    assert torch.count_nonzero(grads["end"]).item()==4
    assert torch.count_nonzero(grads["anchor"]).item()==8


def test_real_video_split_blocks_training_test_leakage():
    ids=[f"human_demos/train/no_distraction/push_cube/{n:04d}/front.mp4"
         for n in range(6)]
    split=video_split(ids,42)
    assert len(split["train"])==4
    assert len(split["val"])==1
    assert len(split["test"])==1
    assert len(set(split["train"]) & set(split["test"]))==0
    with pytest.raises(ValueError,match="FIVE"):
        video_split(ids[:4])


def test_future_labels_are_strictly_after_past_frames():
    n=16
    xy=np.stack([np.arange(n)*.01,np.arange(n)*.002],axis=-1).astype(np.float32)
    z=np.stack([np.full(8,i,dtype=np.float32) for i in range(n)])
    record=dict(video="vA",features=z,centers=xy,native_wh=[512,512])
    split=dict(train=["vA"],val=[],test=[])
    out=build_windows([record],split)
    ex=out["train"][0]
    assert ex["start"]==0
    assert ex["z"].shape==(2,8)
    assert ex["target"].shape==(8,2)
    np.testing.assert_array_equal(ex["z"][:,0],np.array([0,1]))
    np.testing.assert_allclose(ex["target"][0],xy[2])
    np.testing.assert_allclose(ex["target"][-1],xy[9])


def test_persistence_velocity_are_equal_information_baselines():
    observed=np.array([[.30,.2],[.31,.2]],np.float32)
    sample=dict(xy=observed)
    p=baseline_predict(sample,"persistence")
    v=baseline_predict(sample,"constant_velocity")
    assert p.shape==v.shape==(8,2)
    assert np.allclose(p[:,0],.31)
    assert v[7,0]>.31


def test_no_implicit_random_vit_checkpoint_fallback():
    import inspect
    from agcofl import vit_flow_pilot as p
    source=inspect.getsource(p._extract)
    assert "AutoModel.from_pretrained" in source
    assert "requires_grad_(False)" in source
    assert "AutoModel.from_config" not in source
    assert 'device="cuda"' in inspect.signature(p.run).__str__()
