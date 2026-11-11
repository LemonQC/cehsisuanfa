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


def test_local_completeness_and_idempotent_hardlink_cache(tmp_path):
    dest=tmp_path/"download"
    for item in _siblings():
        path=dest/item.rfilename
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b"foo")
    result=p.local_manifest_check(_siblings(),dest)
    assert result["complete"]
    cache=tmp_path/"openpi"
    link=p.install_original_cache((dest/p.PREFIX).resolve(),cache)
    assert Path(link).is_dir() and not Path(link).is_symlink()
    assert (Path(link).resolve()).is_relative_to(cache.resolve())
    assert p.install_original_cache((dest/p.PREFIX).resolve(),cache)==link
    assert (Path(link)/"_METADATA").samefile(dest/p.PREFIX/"_METADATA")
    project=tmp_path/"project"
    manifest=project/"outputs/v120_baselines/openpi_base_hf_manifest.json"
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({
       "source":p.REPO,"revision":p.REVISION,
       "local_dir":str(dest),
       "openpi_data_home":str(cache),
       "openpi_base_cache_link":link,
       "openpi_base_cache_path":link,
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


def test_zero_byte_orbax_success_marker_is_created_and_hash_verified_without_network(
        tmp_path, monkeypatch):
    """Actual pinned manifest has a 0-byte commit_success.txt.

    Offset==size==0 must not mistake a nonexistent .part for a
    downloaded, hashable file. The Git empty-blob digest is verified.
    """
    import hashlib
    from types import SimpleNamespace as NS

    item = NS(
        rfilename=p.PREFIX+"commit_success.txt", size=0,
        blob_id=hashlib.sha1(b"blob 0"+bytes([0])).hexdigest(),
        lfs=None)
    def must_not_download(*args, **kwargs):
        raise AssertionError("No HTTP should be used for verified empty files")
    monkeypatch.setattr(p.requests, "get", must_not_download)
    assert p.download_one(item, tmp_path, "https://huggingface.co", 0) == "git-blob-verified"
    final = tmp_path/item.rfilename
    assert final.is_file()
    assert final.stat().st_size == 0
    assert not final.with_name(final.name+".part").exists()
    # After resume/retry, the completed zero-byte file is reused.
    assert p.download_one(item, tmp_path, "https://huggingface.co", 0) == "git-blob-verified"


def test_zero_byte_orbax_marker_rejects_wrong_hash_without_network(
        tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    item = NS(
        rfilename=p.PREFIX+"commit_success.txt", size=0,
        blob_id="f"*40, lfs=None)
    def must_not_download(*args, **kwargs):
        raise AssertionError("Unexpected HTTP GET")
    monkeypatch.setattr(p.requests, "get", must_not_download)
    with pytest.raises(RuntimeError, match="bad hash"):
        p.download_one(item, tmp_path, "https://huggingface.co", 0)
    assert not (tmp_path/item.rfilename).exists()


def test_old_symlink_migrates_to_hardlinks_and_openpi_resolve_stays_in_cache(tmp_path):
    """Reproduce exact server ValueError and verify no payload byte-copy."""
    import os
    src=tmp_path/"mirror/pi05_base/params"
    (src/"d").mkdir(parents=True)
    (src/"d/shard").write_bytes(b"ORIGINAL large shard stand-in")
    (src/"_METADATA").write_text("{}")
    cache=tmp_path/"openpi_cache"
    dest=cache/"openpi-assets/checkpoints/pi05_base/params"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(src,target_is_directory=True)
    assert not dest.resolve().is_relative_to(cache.resolve())  # original bug
    restored=p.install_original_cache(src,cache)
    assert restored==str(dest)
    assert not dest.is_symlink()
    assert dest.resolve().is_relative_to(cache.resolve())
    assert os.stat(dest/"d/shard").st_ino==os.stat(src/"d/shard").st_ino
    assert os.stat(dest/"d/shard").st_dev==os.stat(src/"d/shard").st_dev
    assert p.install_original_cache(src,cache)==str(dest)
    assert (src/"d/shard").read_bytes()==b"ORIGINAL large shard stand-in"


def test_offline_repair_uses_local_verified_manifest_without_hf_network(tmp_path,monkeypatch):
    import os
    import json
    src=tmp_path/"outputs/v120_baselines/openpi_base_hf_mirror/pi05_base/params"
    (src/"d").mkdir(parents=True)
    (src/"ocdbt.process_0").mkdir()
    for name in ("_METADATA","_CHECKPOINT_METADATA","manifest.ocdbt","d/shard",
                 "ocdbt.process_0/shard"):
        (src/name).write_bytes(b"checked")
    cache=tmp_path/"outputs/v120_baselines/openpi_cache"
    dest=cache/"openpi-assets/checkpoints/pi05_base/params"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(src,target_is_directory=True)
    record_path=tmp_path/"outputs/v120_baselines/openpi_base_hf_manifest.json"
    record_path.write_text(json.dumps({
       "source":p.REPO, "revision":p.REVISION,
       "local_dir":str(src.parent.parent.parent),
       "openpi_data_home":str(cache),
       "openpi_base_cache_link":str(dest),
       "manifest_verification":{"complete":True}}))
    # local_dir must be project outputs/v120_baselines/openpi_base_hf_mirror
    record=json.loads(record_path.read_text())
    record["local_dir"]=str(tmp_path/"outputs/v120_baselines/openpi_base_hf_mirror")
    record_path.write_text(json.dumps(record))
    monkeypatch.setattr(p,"PROJECT",tmp_path)
    def deny_network(*args,**kwargs):
        raise AssertionError("offline repair must not access network")
    monkeypatch.setattr(p.requests,"get",deny_network)
    assert p.main(["--repair-cache-only","--dest",record["local_dir"],
                   "--cache",str(cache)])==0
    assert not dest.is_symlink()
    assert (dest/"_METADATA").samefile(src/"_METADATA")
    assert c.trusted_mirror_install(tmp_path,cache)


def test_old_symlink_is_rejected_by_siglip_cache_gate(tmp_path):
    import json
    src=tmp_path/"download/pi05_base/params"
    src.mkdir(parents=True)
    (src/"_METADATA").write_text("data")
    cache=tmp_path/"cache"
    dest=cache/"openpi-assets/checkpoints/pi05_base/params"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(src,target_is_directory=True)
    manifest=tmp_path/"outputs/v120_baselines/openpi_base_hf_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "source":p.REPO,"revision":p.REVISION,
        "local_dir":str(tmp_path/"download"),
        "openpi_data_home":str(cache),
        "openpi_base_cache_link":str(dest),
        "manifest_verification":{"complete":True}}))
    assert not c.trusted_mirror_install(tmp_path,cache)
