"""Selective original RoboReel pi0.5 assets: only ND held-out PushCube 0080..0099.

No training, no API-based Keyframes, no re-evaluation of Vid2Robot. By default this is a read-only
preflight. All costly actions require separate explicit CLI flags.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .v140_official_pi05 import IDS, VARIANTS, PIN, pinned_sha, preflight, save


def asset_paths(data_root, demo_id):
    root = Path(data_root)
    return (
        root / f"human_demos/eval/no_distraction/push_cube/{demo_id:04d}/front.mp4",
        root / f"video_demonstration_features/without_distraction/push_cube/{demo_id:04d}/front.npy",
        root / f"bpp_keyframe_annotations/without_distraction/push_cube/{demo_id:04d}.json",
    )


def pending(data_root, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    missing = {"video": [], "feature": [], "keyframe": []}
    for i in IDS:
        video, feature, keyframe = asset_paths(data_root, i)
        for label, path in (("video", video), ("feature", feature)):
            if not path.is_file() or path.stat().st_size <= 0:
                missing[label].append(i)
        if variant == "keyframes" and not keyframe.is_file():
            missing["keyframe"].append(i)
    return missing


def prepare_features(official_root, data_root, demo_ids):
    """Call the pinned authors' SigLIP preprocessor on selected MP4s only."""
    if not demo_ids:
        return []
    root = Path(official_root).resolve()
    entry = root / "scripts/compute_vision_features_for_demo.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "src/external_models")]
    spec = importlib.util.spec_from_file_location("_authors_siglip", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = None
    completed = []
    for demo_id in demo_ids:
        video, feature, _ = asset_paths(data_root, demo_id)
        if not video.is_file():
            raise FileNotFoundError(video)
        if feature.is_file() and feature.stat().st_size > 0:
            continue
        rgb = module._read_camera_mp4(video.parent, "front")
        if rgb is None:
            raise RuntimeError(f"Could not decode official video {video}")
        if encoder is None:
            encoder = module._PersistentEncoder(max_T=max(1024, len(rgb)))
        status = encoder.encode_demo_camera(feature.parent, rgb, "front",
                                            overwrite=False)
        if not feature.is_file() or feature.stat().st_size <= 0:
            raise RuntimeError(f"Official encoder failed {feature}: {status}")
        completed.append(demo_id)
        print("PI05_SIGLIP_PREPARED", f"{demo_id:04d}", status, flush=True)
    return completed


def download_checkpoint(official_root, assets, variant):
    root = Path(official_root).resolve()
    entry = root / "scripts/download_checkpoints.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    subprocess.run([sys.executable, str(entry), "--method", VARIANTS[variant],
                    "--condition", "nd", "--dest",
                    str(Path(assets) / "released_checkpoints")],
                   cwd=root, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("uniform",),
                        help="API-free track only; VLM Keyframes deferred")
    parser.add_argument("--official-root", default="third_party/roboreel_models")
    parser.add_argument("--assets", default="outputs/v120_baselines")
    parser.add_argument("--out", default=None)
    parser.add_argument("--download-checkpoint", action="store_true",
                        help="Opt-in large official ND checkpoint download")
    parser.add_argument("--extract-features", action="store_true",
                        help="Opt-in author's SigLIP on only EVAL 0080..0099")
    args = parser.parse_args(argv)
    root, assets = Path(args.official_root).resolve(), Path(args.assets).resolve()
    data_root = assets / "official_eval_data"
    if any((args.download_checkpoint, args.extract_features)):
        if pinned_sha(root) != PIN:
            raise RuntimeError("Only the pinned authors' original commit can prepare assets")
    before = pending(data_root, args.variant)
    if before["video"] and args.extract_features:
        raise FileNotFoundError(f"Frozen eval videos missing: {before['video']}")
    if args.download_checkpoint:
        download_checkpoint(root, assets, args.variant)
    if args.extract_features:
        prepare_features(root, data_root, before["feature"])
    after = pending(data_root, args.variant)
    check = preflight(args.variant, root, assets)
    out = Path(args.out or f"outputs/v141_pi05_{args.variant}")
    save(out / "asset_gate.json", {
        "type": "PI05_SELECTIVE_ASSETS_ONLY_NO_PHYSICS",
        "variant": args.variant,
        "executed_flags": dict(download_checkpoint=args.download_checkpoint,
                               extract_features=args.extract_features),
        "before": before, "after": after,
        "official_preflight": check,
        "physics_executed": False, "trained": False,
        "note": "Evaluation-time demo feature extraction is not training; EVAL cannot tune model parameters."})
    print("PI05_SELECTIVE_ASSET_GATE", json.dumps({
        "variant": args.variant, "ready": check["ready_for_runtime"],
        "missing": after, "failures": check["failures"]}), flush=True)


if __name__ == "__main__":
    main()
