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
