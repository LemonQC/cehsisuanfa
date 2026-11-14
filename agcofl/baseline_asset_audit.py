"""Read-only local asset/runtime audit for TRO1 B11 Im2Flow2Act and B12 EC-Flow.

This script never downloads, installs, extracts, edits, or launches a simulator.
It reports what is already present so native reproduction can be planned without
silently mixing source tasks, weights, or environments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "B11": {
        "name": "Im2Flow2Act",
        "expected_commit": "f44b368b76195c30bf38074f6419fc8b9f8feb5f",
        "repo": "https://github.com/real-stanford/im2Flow2Act",
        "required": [
            "LICENSE",
            "environment.yml",
            "scripts/flow_generation/inference.py",
            "scripts/controll/evaluate_from_flow_generation.py",
            "im2flow2act/diffusion_policy/utility/evaluation.py",
            "config/flow_generation/inference.yaml",
            "config/diffusion_policy/evaluate_from_flow_generation.yaml",
        ],
        "asset_groups": {
            "flow_generation_checkpoint": [
                "checkpoints/flow_generation/.hydra/config.yaml",
                "checkpoints/flow_generation/epoch=1800.ckpt",
            ],
            "manipulation_policy_checkpoint": [
                "checkpoints/manipulation_policy/.hydra/config.yaml",
                "checkpoints/manipulation_policy/epoch=300.ckpt",
                "checkpoints/manipulation_policy/stats.pickle",
            ],
            "stable_diffusion_pretrained": [
                "pretrain_weights/StableDiffusion1.5",
            ],
            "native_simulation_eval_data": [
                "data/simulation_evaluation/drawer_open",
                "data/simulation_evaluation/pickNplace",
                "data/simulation_evaluation/pouring",
            ],
            "camera_calibration": [
                "data_local/simulation/instrinsic_5-1.pkl",
                "data_local/simulation/pose_matrix_5-1.pkl",
            ],
        },
        "python_modules": [
            "torch", "torchvision", "diffusers", "hydra", "zarr",
            "jax", "cv2", "faiss",
        ],
        "notes": [
            "Native policy is built around the authors' UR5e/MuJoCo stack and 7-D action space.",
            "The official end-to-end evaluator couples online point tracking and policy inference; README states >=24 GB GPU memory.",
            "RoboReel Panda wristcam / pd_joint_pos 8-D action is not a drop-in interface.",
            "Do not train on RoboReel or report RoboReel scores until a separately documented action/camera/robot port exists.",
        ],
    },
    "B12": {
        "name": "EC-Flow",
        "expected_commit": "7d0a6243fd8b92500b5f864d70b3b60d6b6c853a",
        "repo": "https://github.com/YixiangChen515/EC-Flow",
        "required": [
            "LICENSE",
            "install.sh",
            "inference.py",
            "experiment/eval_my_flow_policy.py",
            "experiment/eval_policy.sh",
            "experiment/policy.py",
            "metaworld",
        ],
        "asset_groups": {
            "flow_checkpoint": ["ckpt/flow.pt"],
            "goal_image_checkpoint": ["ckpt/goal_img.pt"],
            "sam2_checkpoint": ["sam_and_track/checkpoints/sam2.1_hiera_large.pt"],
            "grounding_dino_checkpoint": ["sam_and_track/gdino_checkpoints/groundingdino_swint_ogc.pth"],
            "co_tracker_checkpoint": ["sam_and_track/co-tracker/checkpoints/scaled_offline.pth"],
        },
        "python_modules": [
            "torch", "torchvision", "diffusers", "clip", "mujoco",
            "metaworld", "cv2", "imageio",
        ],
        "notes": [
            "Native benchmark is Meta-World Sawyer, not RoboReel Panda.",
            "Authors describe 8x RTX 4090 for training; prioritize released-weight native inference, not full retraining.",
            "Original eval script treats episode length <=500 as success proxy; audit and report this separately from Meta-World env info['success'] before interpreting numbers.",
            "SAM2, GroundingDINO, and CoTracker checkpoints are additional required assets for the released pipeline.",
            "RoboReel adaptation requires a new camera/robot/action/control bridge; do not label it official native reproduction.",
        ],
    },
}

CANDIDATES = {
    "B11": [
        "CrossPaper_Adapters_v4/im2flow2act",
        "third_party/im2Flow2Act",
        "third_party/im2flow2act",
        "external/im2Flow2Act",
    ],
    "B12": [
        "CrossPaper_Adapters_v4/EC-Flow",
        "third_party/EC-Flow",
        "third_party/ec-flow",
        "external/EC-Flow",
    ],
}


def git_info(path: Path) -> dict:
    if not (path / ".git").exists():
        return {"is_git_repo": False, "commit": None, "dirty": None}
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), *args],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            return None
    return {
        "is_git_repo": True,
        "commit": run(["rev-parse", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
    }


def path_state(root: Path, rel: str) -> dict:
    p = root / rel
    exists = p.exists()
    if p.is_file():
        size = p.stat().st_size
        file_count = 1
    elif p.is_dir():
        target = p.resolve()
        files = [x for x in target.rglob("*") if x.is_file()]
        size = sum(x.stat().st_size for x in files)
        file_count = len(files)
    else:
        size = None
        file_count = 0
    return {"path": rel, "exists": exists, "kind": "file" if p.is_file() else "dir" if p.is_dir() else "missing", "bytes": size, "file_count": file_count}


def inspect_method(method: str, repo_root: Path, explicit_root: Path | None) -> dict:
    spec = SPECS[method]
    candidates = [explicit_root] if explicit_root else [repo_root / x for x in CANDIDATES[method]]
    local = next((p.resolve() for p in candidates if p and p.exists()), None)
    if local is None:
        return {
            "id": method, "name": spec["name"], "source_url": spec["repo"],
            "expected_commit": spec["expected_commit"], "local_root": None,
            "source_status": "SOURCE_NOT_PRESENT_LOCALLY",
            "required_files": [], "asset_groups": {},
            "missing_required": spec["required"],
            "asset_groups_ready": {}, "python_modules": {},
            "notes": spec["notes"],
        }
    required = [path_state(local, x) for x in spec["required"]]
    assets = {
        group: [path_state(local, x) for x in paths]
        for group, paths in spec["asset_groups"].items()
    }
    group_ready = {
        group: all(x["exists"] and x["file_count"] > 0 and (x["bytes"] or 0) > 0 for x in entries)
        for group, entries in assets.items()
    }
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in spec["python_modules"]
    }
    gi = git_info(local)
    commit = gi["commit"]
    if not all(x["exists"] for x in required):
        source_status = "SOURCE_TREE_INCOMPLETE"
    elif commit and commit != spec["expected_commit"]:
        source_status = "SOURCE_COMMIT_MISMATCH"
    elif commit is None:
        source_status = "SOURCE_PRESENT_COMMIT_UNVERIFIED"
    elif gi["dirty"]:
        source_status = "SOURCE_PINNED_DIRTY"
    else:
        source_status = "SOURCE_PINNED_CLEAN"
    return {
        "id": method, "name": spec["name"], "source_url": spec["repo"],
        "expected_commit": spec["expected_commit"], "local_root": str(local),
        "source_status": source_status, "git": gi,
        "required_files": required,
        "missing_required": [x["path"] for x in required if not x["exists"]],
        "asset_groups": assets, "asset_groups_ready": group_ready,
        "python_modules": modules, "notes": spec["notes"],
    }


def runtime_info(root: Path) -> dict:
    usage = shutil.disk_usage(root)
    nvidia = None
    try:
        nvidia = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        ).strip().splitlines()
    except Exception as exc:
        nvidia = {"error": str(exc)}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(root),
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "nvidia_smi": nvidia,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--b11-root", type=Path, default=None)
    parser.add_argument("--b12-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    report = {
        "schema": "TRO1_B11_B12_ASSET_AUDIT_V1",
        "read_only": True,
        "runtime": runtime_info(root),
        "methods": {
            "B11": inspect_method("B11", root, args.b11_root),
            "B12": inspect_method("B12", root, args.b12_root),
        },
        "decision": "AUDIT_ONLY_NO_DOWNLOAD_NO_INSTALL_NO_TRAIN_NO_PHYSX",
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        out = args.out if args.out.is_absolute() else root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise SystemExit(f"REFUSE_OVERWRITE_EXISTING_AUDIT: {out}")
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"TRO1_B11_B12_AUDIT_SAVED {out}")


if __name__ == "__main__":
    main()
