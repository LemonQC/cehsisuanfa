"""Offline fixture tests; no Orbax tensor reads or network."""
import json
from scripts import inspect_pi05_uniform_vision_metadata as m


def test_metadata_detects_candidate_without_ever_proving_equivalence(tmp_path):
    root=tmp_path/"ckpt"
    params=root/"params"
    params.mkdir(parents=True)
    (params/"_METADATA").write_text(json.dumps({
        "tree_metadata":{
            "('PaliGemma', 'img', 'embedder')":{"key_metadata":[]},
            "('PaliGemma', 'llm', 'weights')":{"key_metadata":[]}
        }
    }))
    (tmp_path/"train_config.json").write_text(
        json.dumps({"tasks":["push_cube"],"demonstration_config":"perceptual"}))
    r=m.audit(root)
    assert r["orbax_metadata"]["paligemma_img_key_candidates"]==1
    assert r["orbax_metadata"]["strict_vision_weights_equivalence_proven"] is False
    assert r["tensors_loaded"] is False
    assert r["network_used"] is False


def test_no_metadata_is_safe_failure(tmp_path):
    root=tmp_path/"ckpt"
    (root/"params").mkdir(parents=True)
    assert m.metadata_keys(root)["found"] is False
    assert m.main(["--checkpoint",str(root)]) == 2


def test_corrupt_metadata_is_not_treated_as_checkpoint(tmp_path):
    root=tmp_path/"ckpt"
    (root/"params").mkdir(parents=True)
    (root/"params"/"_METADATA").write_bytes(b"not-json")
    r=m.metadata_keys(root)
    assert r["found"] is True
    assert "unreadable" in r["error"]
