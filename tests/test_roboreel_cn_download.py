"""Offline tests only. Must not contact Hugging Face."""
from pathlib import Path
import pytest
from agcofl.download_roboreel_cn import (
    video_relative_path, valid_video, ensure_video
)
from agcofl.roboreel_bridge import evaluate


def test_one_video_known_official_path():
    assert video_relative_path(80) == (
      'human_demos/eval/no_distraction/push_cube/0080/front.mp4')
    assert video_relative_path(99, "left").endswith('/0099/left.mp4')
    with pytest.raises(ValueError):
        video_relative_path(79)


def test_existing_local_mp4_requires_no_download(monkeypatch,tmp_path):
    p=tmp_path/video_relative_path(80)
    p.parent.mkdir(parents=True)
    p.write_bytes(b'\x00\x00\x00\x18ftypisom' + b'\x00'*1024)
    assert valid_video(p)
    # If already local, no import of Hugging Face or network required.
    assert ensure_video(tmp_path,80)==p


def test_invalid_existing_file_is_not_silently_accepted(tmp_path):
    p=tmp_path/video_relative_path(80)
    p.parent.mkdir(parents=True)
    p.write_bytes(b'HTML 503 Bad Gateway')
    assert not valid_video(p)
    with pytest.raises(RuntimeError,match='Invalid existing MP4'):
        ensure_video(tmp_path)


def test_no_human_video_diagnostic_rejects_video_conditioning():
    with pytest.raises(ValueError,match='--no-video requires'):
        evaluate(root='nonexistent',episodes=[0],out='unused',
                 method='video_motion',no_video=True)
