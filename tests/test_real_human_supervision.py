"""Tests are entirely offline: no Hugging Face network and no ManiSkill renderer."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from agcofl.real_human_supervision import (
    HumanMaskPredictor, make_windows, split_video_ids, video_masks
)
from agcofl.physics_replay import render_trace


def test_real_future_windows_use_only_past_mask_frames():
    array = np.zeros((18,32,32),np.float32)
    for i in range(len(array)):
        array[i,15:18,2+i:5+i]=1.
    windows=make_windows(array,"0080",stride=4)
    assert len(windows)==3
    ex=windows[0]
    np.testing.assert_array_equal(ex["past"],array[:2])
    np.testing.assert_array_equal(ex["future"],array[2:10])
    # No future frame passed into the model input.
    assert ex["past"].shape==(2,32,32)
    assert ex["future"].shape==(8,32,32)
    assert not np.array_equal(ex["past"][0], ex["future"][0])  # distinct observed and future frames
    assert ex["source_id"]=="0080"


def test_train_only_when_less_than_five_independent_videos():
    split,status=split_video_ids(["0080"],seed=42)
    assert status=="TRAIN_ONLY_NOT_GENERALIZATION"
    assert split==dict(train=["0080"],val=[],test=[])
    split,status=split_video_ids([str(i) for i in range(80,86)],seed=42)
    assert status=="HELD_OUT_HUMAN_VIDEOS_INTERNAL_ONLY"
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])


def test_mask_predictor_gradient_from_real_mask_input():
    net=HumanMaskPredictor()
    past=torch.zeros(2,2,32,32);past[:,0,8:12,7:11]=1.
    pred=net(past)
    assert pred.shape==(2,8,32,32)
    pred.square().mean().backward()
    assert net.net[0].weight.grad is not None
    assert torch.isfinite(net.net[0].weight.grad).all()


def test_decode_authentic_format_mp4_with_blue_cube(tmp_path):
    video=tmp_path/"front.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),12,(128,128))
    if not writer.isOpened():
        pytest.skip("mp4v writer not available")
    for i in range(28):
        frame=np.zeros((128,128,3),np.uint8)
        # Blue in BGR (OpenCV camera)
        frame[50:78,20+i:48+i]=(220,45,15)
        writer.write(frame)
    writer.release()
    masks,meta=video_masks(video,frames=24)
    assert masks.shape==(24,32,32)
    assert meta["mask_coverage"]>.8
    assert np.sum(masks)>0
    assert len(make_windows(masks,"0080"))>0


def test_physics_replay_from_saved_real_state_schema(tmp_path):
    episode=tmp_path/"episode_0000"
    episode.mkdir()
    trace=[]
    for i in range(6):
        trace.append(dict(phase="push",cube=[-.1+i*.03,.01,.02],
                          tcp=[-.15+i*.03,.01,.03],
                          goal_distance_m=.1,attempt=0,action=[0]*7))
    (episode/"physical_trace.json").write_text(json.dumps(trace))
    (episode/"episode_result.json").write_text(json.dumps({
        "success":True,"final_cube":[.08,.01,.02],
        "final_goal":[.09,.01,.001],
        "final_goal_error_m":.01
    }))
    mp4=render_trace(episode)
    assert mp4.exists() and mp4.stat().st_size>512
    result=json.loads(mp4.with_suffix(".json").read_text())
    assert result["label"]=="PHYSICS_XY_TRACE_REPLAY_NOT_RGB_SIMULATION"
    assert result["frames"]==7


def test_real_mp4_three_mode_training_single_video_is_labeled_train_only(tmp_path):
    from agcofl.real_human_supervision import train_real_videos

    root=tmp_path/"videos"/"0080"
    root.mkdir(parents=True)
    source=root/"front.mp4"
    w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),12,(96,96))
    if not w.isOpened():
        pytest.skip("No MP4 encoder")
    for i in range(32):
        frame=np.zeros((96,96,3),np.uint8)
        frame[34:57,12+i:34+i]=(220,45,15)
        w.write(frame)
    w.release()

    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        out=tmp_path/"train_output"
        report=train_real_videos(root.parent,out,epochs=1,seed=42,
                                 max_videos=1,device="cpu",frames=24,batch=4)
    finally:
        torch.set_num_threads(old_threads)

    assert report["validity"]=="TRAIN_ONLY_NOT_GENERALIZATION"
    assert report["test_windows"]==0
    assert report["train_windows"]>0
    assert report["modes"]==["full","anchor","end"]
    for mode in ("full","anchor","end"):
        assert (out/f"{mode}_real_video.pt").is_file()
    assert (out/"mode_summary.csv").exists()
    assert (out/"three_mode_predictions.png").exists()
    assert (out/"video_0080"/"human_mask_overlay.png").exists()
