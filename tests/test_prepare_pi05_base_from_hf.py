"""Offline tests for pinned HF OpenPI base manifest and GCS cache placement."""
from pathlib import Path
from types import SimpleNamespace as NS
import json

import pytest
from scripts import prepare_pi05_base_from_hf as p
from scripts import check_pi05_siglip_cache as c


def _siblings():
    return [NS(rfilename=p.PREFIX+x,size=3) for x in (
        "_METADATA", "_CHECKPOINT_METADATA", "manifest.ocdbt",
        "d/data", "ocdbt.process_0/bucket")]


def test_only_pin_original_openpi_params_subtree():
    selected=p.selection(_siblings()+[
        NS(rfilename="pi05_droid/params/_METADATA",size=10),
        NS(rfilename="pi05_base/README.md",size=10)])
    assert len(selected)==5
    assert all(x.rfilename.startswith(p.PREFIX) for x in selected)


def test_fail_missing_orbax_metadata():
    with pytest.raises(RuntimeError):
        p.selection([NS(rfilename=p.PREFIX+"d/data",size=1)])


def test_local_completeness_and_idempotent_symlink(tmp_path):
    dest=tmp_path/"download"
    for item in _siblings():
        path=dest/item.rfilename
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b"foo")
    result=p.local_manifest_check(_siblings(),dest)
    assert result["complete"]
    cache=tmp_path/"openpi"
    link=p.link_original_cache((dest/p.PREFIX).resolve(),cache)
    assert Path(link).is_symlink()
    assert p.link_original_cache((dest/p.PREFIX).resolve(),cache)==link
    project=tmp_path/"project"
    manifest=project/"outputs/v120_baselines/openpi_base_hf_manifest.json"
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({
       "source":p.REPO,"revision":p.REVISION,
       "openpi_data_home":str(cache),
       "openpi_base_cache_link":link,
       "manifest_verification":result}))
    assert c.trusted_mirror_install(project,cache)
    (dest/p.PREFIX/"d/data").unlink()
    assert not p.local_manifest_check(_siblings(),dest)["complete"]


def test_incomplete_install_not_trusted(tmp_path):
    assert not c.trusted_mirror_install(tmp_path,tmp_path/"cache")


def test_http_get_skips_hub_head_and_verifies_digest(tmp_path, monkeypatch):
    import hashlib
    import io
    from types import SimpleNamespace as NS
    content=b"test orbax metadata"
    expected=NS(rfilename=p.PREFIX+"_METADATA",size=len(content),
                lfs={"sha256":hashlib.sha256(content).hexdigest()},blob_id=None)
    seen={}
    class Reply:
        url="https://cdn-lfs.huggingface.co/verified-blob"
        status_code=200
        headers={}
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def raise_for_status(self): pass
        def iter_content(self,chunk_size): yield content
    def fake_get(url,**kwargs):
        seen["url"]=url
        seen["kwargs"]=kwargs
        return Reply()
    monkeypatch.setattr(p.requests,"get",fake_get)
    state=p.download_one(expected,tmp_path,"https://huggingface.co",0)
    assert state=="sha256-verified"
    assert (tmp_path/expected.rfilename).read_bytes()==content
    assert seen["kwargs"]["stream"] is True
    assert p.REVISION in seen["url"]
    assert not (tmp_path/(expected.rfilename+".part")).exists()


def test_get_206_resumes_exact_offset_without_duplicate(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace as NS
    content=b"hello-world"
    expected=NS(rfilename=p.PREFIX+"d/shard0",size=len(content),
                lfs={"sha256":hashlib.sha256(content).hexdigest()},blob_id=None)
    part=tmp_path/(expected.rfilename+".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(content[:5])
    class Reply:
        url="https://cdn-lfs.huggingface.co/verified-blob"
        status_code=206
        headers={"Content-Range":f"bytes 5-10/{len(content)}"}
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def raise_for_status(self): pass
        def iter_content(self,chunk_size): yield content[5:]
    def fake_get(url,**kwargs):
        assert kwargs["headers"]["Range"]=="bytes=5-"
        return Reply()
    monkeypatch.setattr(p.requests,"get",fake_get)
    assert p.download_one(expected,tmp_path,"https://huggingface.co",0)=="sha256-verified"
    assert (tmp_path/expected.rfilename).read_bytes()==content


def test_git_blob_sha1_uses_zero_byte_header(tmp_path):
    import hashlib
    from types import SimpleNamespace as NS
    body=b"tiny"
    name=p.PREFIX+"_METADATA"
    file=tmp_path/name
    file.parent.mkdir(parents=True)
    file.write_bytes(body)
    h=hashlib.sha1(b"blob 4"+bytes([0])+body).hexdigest()
    sib=NS(rfilename=name,size=4,lfs=None,blob_id=h)
    assert p.file_state(file,sib)=="git-blob-verified"
