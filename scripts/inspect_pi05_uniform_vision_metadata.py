#!/usr/bin/env python3
"""Read-only inspection of original RoboReel ND Uniform Orbax metadata.

No JAX model restore, no unpickling, no tensor reads, no network, no CUDA.
This answers only whether the *released fine-tuned* checkpoint appears to
contain a PaliGemma.img subtree; it does NOT prove that this is equivalent
to the OpenPI pi05_base vision encoder required by original SigLipTokenizer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path("outputs/v120_baselines/released_checkpoints/nd/vid_cond_pi05_uniform/ckpt")
MAX_METADATA = 32 * 1024 * 1024


def metadata_keys(ckpt: Path):
    loc=ckpt / "params" / "_METADATA"
    if not loc.is_file():
        return dict(found=False, error="No Orbax params/_METADATA", path=str(loc))
    if loc.stat().st_size > MAX_METADATA:
        return dict(found=True, error="Metadata too large to inspect safely", path=str(loc))
    try:
        meta=json.loads(loc.read_text())
    except (ValueError, UnicodeError) as exc:
        return dict(found=True,error="Orbax metadata JSON unreadable: "+type(exc).__name__,path=str(loc))
    tree=meta.get("tree_metadata",{}) if isinstance(meta,dict) else {}
    items=[]
    if isinstance(tree,dict):
        items=list(tree)
    # Orbax saves path keys in nested tuple encodings; report only key names,
    # never assume matching text alone establishes a valid tensor subtree.
    segments=[name for name in items if "PaliGemma" in name or "img" in name]
    return dict(found=True,path=str(loc),top_level=list(meta)[:20],
                tree_entries=len(items),
                paligemma_img_key_candidates=len([name for name in segments
                  if "PaliGemma" in name and "img" in name]),
                sample_tree_keys=segments[:8],
                strict_vision_weights_equivalence_proven=False)


def audit(root: Path):
    train_config = root.parent / "train_config.json"
    if not train_config.is_file():
        train_config=root / "train_config.json"
    conf={}
    if train_config.is_file():
        try:
            c=json.loads(train_config.read_text())
            conf={key:c.get(key) for key in (
                "tasks","task","demonstration_config","model","freeze_vision",
                "freeze_img","trainable_filter") if key in c}
        except (ValueError,UnicodeError):
            conf={"config_read_error":True}
    params=root / "params"
    return dict(type="PI05_UNIFORM_VISION_SUBTREE_READ_ONLY",
        checkpoint_dir=str(root), checkpoint_params_exists=params.is_dir(),
        train_config_exists=train_config.is_file(),train_config_selected_fields=conf,
        orbax_metadata=metadata_keys(root),
        only_original_base_vision_encoder_qualifies_as_official=True,
        potential_finetuned_checkpoint_vision_not_equivalent_by_assumption=True,
        tensors_loaded=False, network_used=False, runtime_executed=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args(argv)
    report=audit(args.checkpoint.resolve())
    print("PI05_UNIFORM_VISION_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
    if not report["checkpoint_params_exists"] or not report["orbax_metadata"]["found"]:
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
