"""Offline regressions for RoboReel Vid2Robot resumable official asset transfer.

No network requests, downloaded video, pretrained weights or private data.
"""
from types import SimpleNamespace
import pytest

from agcofl.resumable_hf_assets import (
    _valid,_direct_resolve,_request_download,sha256_file
)


def test_reject_unknown_repo_or_cross_split_source():
    good=dict(path="nd/vid2robot/ckpt.pt",bytes=2_368_874_616)
    assert _valid(good,"LogosRoboticsLab/roboreel_models","model","https://hf-mirror.com")[0]>0
    with pytest.raises(ValueError,match="allowlist"):
        _valid(good,"untrusted/model","model","https://hf-mirror.com")
    with pytest.raises(ValueError,match="Only official PushCube"):
        _valid(dict(path="human_demos/train/no_distraction/push_cube/0000/front.mp4",bytes=3000),
               "LogosRoboticsLab/roboreel","dataset","https://hf-mirror.com")
    with pytest.raises(ValueError,match="Unsafe"):
        _valid(dict(path="../escape",bytes=1000),
               "LogosRoboticsLab/roboreel_models","model","https://hf-mirror.com")
    with pytest.raises(ValueError,match="endpoint"):
        _valid(good,"LogosRoboticsLab/roboreel_models","model","https://example.com")


def test_official_dataset_and_model_resolve_paths_are_not_swapped():
    model=_direct_resolve("LogosRoboticsLab/roboreel_models","model",
          "https://hf-mirror.com","nd/vid2robot/train_config.json")
    data=_direct_resolve("LogosRoboticsLab/roboreel","dataset",
          "https://hf-mirror.com","human_demos/eval/no_distraction/push_cube/0080/front.mp4")
    assert "/datasets/" not in model
    assert model.endswith("/nd/vid2robot/train_config.json")
    assert "/datasets/LogosRoboticsLab/roboreel/resolve/main/" in data


def test_resume_range_download_with_mock_http(tmp_path,monkeypatch):
    target=tmp_path/"foo.bin"
    part=tmp_path/"foo.bin.part"
    part.write_bytes(b"abc")
    history=[]
    class Response:
        status_code=206
        headers={"Content-Range":"bytes 3-5/6"}
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def iter_content(self,chunk_size):
            yield b"def"
    def fake_get(url,headers,timeout,stream,allow_redirects):
        history.append(headers)
        return Response()
    import sys
    monkeypatch.setitem(sys.modules,"requests",SimpleNamespace(get=fake_get))
    result=_request_download("https://hf-mirror.com/any",target,6,attempts=1)
    assert result==target
    assert target.read_bytes()==b"abcdef"
    assert not part.exists()
    assert history==[{"Range":"bytes=3-"}]
    assert len(sha256_file(target))==64


def test_ignored_http_range_restarts_without_concatenation(tmp_path,monkeypatch):
    target=tmp_path/"foo.bin"
    part=tmp_path/"foo.bin.part"
    # A server that ignores Range must RESTART from zero, not append.
    part.write_bytes(b"old")
    class Response:
        status_code=200
        headers={}
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def iter_content(self,chunk_size):yield b"correct"
    import sys
    monkeypatch.setitem(sys.modules,"requests",SimpleNamespace(
        get=lambda *a,**k:Response()))
    _request_download("https://hf-mirror.com/any",target,7,attempts=1)
    assert target.read_bytes()==b"correct"
