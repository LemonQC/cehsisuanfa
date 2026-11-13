#!/usr/bin/env python3
"""Read-only RoboReel TRAIN H5+dual-MP4 inventory for B01/B02 shared data split.

No network, training, policy, simulation, GPU, model or EVAL access. This is a
DATA/SCHEMA gate, not a claim that B01 Transformer BC or B02 DP is reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "outputs/v110_baselines/official_dataset/robot_trajectories/no_distraction/push_cube"
DEFAULT_OUT = PROJECT / "outputs/research_runs/b01_b02_shared_robot_train"
REQUIRED_VIDEOS = ("base_camera.mp4", "hand_camera.mp4")


def _regular_file(path: Path) -> bool:
    """Disallow symlinked files and directories, including all ancestors."""
    if not path.is_file():
        return False
    return not any(p.is_symlink() for p in (path, *path.parents))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_mp4(path: Path) -> bool:
    with path.open("rb") as source:
        header = source.read(12)
    return len(header) == 12 and header[4:8] == b"ftyp"


def _h5_actions(path: Path) -> tuple[list[dict], str | None]:
    try:
        import h5py
    except ImportError:
        return [], "H5PY_NOT_INSTALLED"
    found: list[dict] = []
    try:
        with h5py.File(path, "r") as handle:
            def inspect(name, value):
                if (isinstance(value, h5py.Dataset) and
                        name.split("/")[-1].lower() in ("action", "actions") and
                        len(value.shape) >= 1 and value.shape[0] > 0):
                    found.append({"h5_path": name, "shape": list(value.shape),
                                  "dtype": str(value.dtype)})
            handle.visititems(inspect)
    except (OSError, ValueError, RuntimeError):
        return [], "H5_UNREADABLE_OR_CORRUPT"
    return sorted(found, key=lambda x: x["h5_path"]), None if found else "ACTION_DATASET_NOT_FOUND"


def inspect_training_data(root: Path, *, required_demos: int = 64, seed: int = 42) -> dict:
    """An offline inventory; never reads privileged held-out demo IDs 0080-0099."""
    if required_demos < 2:
        raise ValueError("REQUIRED_DEMOS_MUST_BE_AT_LEAST_TWO")
    if not 0 <= seed < 2**32:
        raise ValueError("INVALID_RANDOM_SEED")
    root = root.expanduser().absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        return {"schema": "TRO1_B01_B02_TRAIN_DATA_V1", "status": "TRAIN_ROOT_MISSING_OR_UNSAFE",
                "required_demos": required_demos, "valid_demos": 0, "train": [], "validation": [],
                "rejected": [], "records": [], "new_physx_rollouts": 0, "new_training_runs": 0}

    records: list[dict] = []
    rejected: list[dict] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        demo_id = folder.name
        # RoboReel reserves IDs 0080-0099 for held-out EVAL; no reads of those files.
        if not (len(demo_id) == 4 and demo_id.isascii() and demo_id.isdigit() and
                int(demo_id) < 80):
            rejected.append({"demo_id": demo_id, "reason": "NOT_TRAIN_ID"})
            continue
        h5s = sorted(p for p in folder.iterdir()
                     if p.name.endswith(".h5") and ".state." not in p.name)
        chosen = h5s + [folder / name for name in REQUIRED_VIDEOS]
        if len(h5s) != 1 or not all(_regular_file(p) for p in chosen):
            rejected.append({"demo_id": demo_id, "reason": "INCOMPLETE_OR_AMBIGUOUS_TRIPLET"})
            continue
        if not all(_valid_mp4(folder / name) for name in REQUIRED_VIDEOS):
            rejected.append({"demo_id": demo_id, "reason": "INVALID_MP4_HEADER"})
            continue
        action_sets, error = _h5_actions(h5s[0])
        if error:
            rejected.append({"demo_id": demo_id, "reason": error})
            continue
        records.append({"demo_id": demo_id,
                        "files": [{"name": p.name, "bytes": p.stat().st_size,
                                   "sha256": _sha256(p)} for p in chosen],
                        "action_dataset_candidates": action_sets})
    rng = random.Random(seed)
    all_ids = [r["demo_id"] for r in records]
    rng.shuffle(all_ids)
    split_at = min(len(all_ids) - 1, int(len(all_ids) * .8)) if len(all_ids) >= 2 else 0
    train = sorted(all_ids[:split_at]); val = sorted(all_ids[split_at:])
    status = ("TRAIN_SCHEMA_AND_SPLIT_READY_NOT_POLICY_TRAINED"
              if len(records) >= required_demos and len(train) > 0 and len(val) > 0
              else "INSUFFICIENT_AUTHENTIC_TRAIN_TRIPLETS")
    return {"schema": "TRO1_B01_B02_TRAIN_DATA_V1", "status": status,
            "source_claim": "LOCAL_FILES_ONLY_UPSTREAM_PROVENANCE_MUST_BE_AUDITED",
            "task": "push_cube", "split_seed": seed, "required_demos": required_demos,
            "valid_demos": len(records), "train": train, "validation": val,
            "records": sorted(records, key=lambda x: x["demo_id"]),
            "rejected": rejected, "train_scope": "ONLY_DEMO_IDS_0000_TO_0079",
            "not_verified": ["OFFICIAL_DATA_PROVENANCE", "ACTION_NORMALIZATION", "RGB_ACTION_ALIGNMENT",
                             "MODEL_TRAINING", "CHECKPOINT", "ROBOREEL_PHYSX_EVALUATION"],
            "new_physx_rollouts": 0, "new_training_runs": 0}


def write_immutable(root: Path, out_dir: Path, manifest: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    uid = hashlib.sha256(raw).hexdigest()[:16]
    target = out_dir / f"train_manifest_{uid}.json"
    try:
        with target.open("xb") as f:
            f.write(raw)
    except FileExistsError:
        if target.is_symlink() or target.read_bytes() != raw:
            raise RuntimeError("IMMUTABLE_REPORT_COLLISION")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--required-demos", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = inspect_training_data(args.data_root, required_demos=args.required_demos, seed=args.seed)
    output = write_immutable(args.data_root, args.out, report)
    print("TRO1_B01_B02_SHARED_TRAIN_GATE", json.dumps({
        "status": report["status"], "valid_demos": report["valid_demos"],
        "train_count": len(report["train"]), "validation_count": len(report["validation"]),
        "rejected_count": len(report["rejected"]), "report": str(output),
        "new_physx_rollouts": 0, "new_training_runs": 0}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "TRAIN_SCHEMA_AND_SPLIT_READY_NOT_POLICY_TRAINED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
