"""Offline tests: no network/large model transfer occurs."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recover_pi05_uniform_missing as rec


def sib(payload: bytes):
    return SimpleNamespace(size=len(payload),
                           lfs={"sha256": hashlib.sha256(payload).hexdigest()},
                           blob_id=None)


def test_only_missing_or_checksum_bad_is_planned(tmp_path):
    data = b"orbax-checkpoint"
    s = sib(data)
    assert rec.check(tmp_path / "missing", s) == "missing"
    file = tmp_path / "present"
    file.write_bytes(data)
    assert rec.check(file, s) == "verified"
    file.write_bytes(b"x" * len(data))
    assert rec.check(file, s) == "bad-checksum"


def test_git_blob_sha1_is_not_plain_file_sha1(tmp_path):
    content = b"training-config"
    h = hashlib.sha1()
    h.update(("blob %d\\0" % len(content)).encode().replace(b"\\0", b"\x00"))
    h.update(content)
    s = SimpleNamespace(size=len(content), blob_id=h.hexdigest(), lfs=None)
    file = tmp_path / "train_config.json"
    file.write_bytes(content)
    assert rec.check(file, s) == "verified"


def test_resume_range_206_and_atomic_final(monkeypatch, tmp_path):
    content = b"abcdef"
    s = sib(content)
    name = "nd/vid_cond_pi05_uniform/ckpt/params/piece"
    file = tmp_path / name
    file.parent.mkdir(parents=True)
    file.with_name(file.name + ".part").write_bytes(b"abc")

    class Response:
        status_code = 206
        url = "https://huggingface.co/some/path"
        headers = {"Content-Range": "bytes 3-5/6"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b"def"

    def fake_get(url, **kw):
        assert kw["headers"]["Range"] == "bytes=3-"
        assert kw["stream"] is True
        return Response()

    monkeypatch.setattr(rec.requests, "get", fake_get)
    assert rec._download_one(tmp_path, name, s, 0)[1] == "verified"
    assert file.read_bytes() == content
    assert not file.with_name(file.name + ".part").exists()


def test_no_download_if_existing_verified(monkeypatch, tmp_path):
    data = b"cached-valid"
    s = sib(data)
    name = "nd/vid_cond_pi05_uniform/train_config.json"
    file = tmp_path / name
    file.parent.mkdir(parents=True)
    file.write_bytes(data)
    monkeypatch.setattr(rec.requests, "get",
                        lambda *args, **kwargs: pytest.fail("should skip"))
    assert rec._download_one(tmp_path, name, s, 0)[1] == "skip-existing"
