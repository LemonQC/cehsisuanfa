"""Fail-closed RoboReel official video-conditioned pi0.5 ND evaluation.

The authors' demo_pi05 runner uses action_horizon=20. NEVER copy
Vid2Robot's chunk_size=1. All EVAL 0080..0099 are held out for testing only.
No claim of physics execution follows from a successful asset preflight.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PIN = "79a0fb0099e73c5560114753a8706276f93aefc2"
SEEDS = (0, 100000, 200000)
IDS = list(range(80, 100))
CELL = "demo_distract=False|env_distract=False"
VARIANTS = {"uniform": "vid_cond_pi05_uniform", "keyframes": "vid_cond_pi05_keyframes"}
BUCKET = "without_distraction"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


def pinned_sha(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def preflight(variant, official_root, assets):
    if variant not in VARIANTS:
        raise ValueError("Unknown pi0.5 variant")
    root, assets = Path(official_root).resolve(), Path(assets).resolve()
    data, croot = assets / "official_eval_data", assets / "released_checkpoints"
    ckpt = croot / "nd" / VARIANTS[variant] / "ckpt"
    original = root / "configs/eval/nd" / (VARIANTS[variant] + ".json")
    checks = {}
    def gate(key, ok, detail=None):
        checks[key] = {"ok": bool(ok), "detail": detail}

    sha = pinned_sha(root)
    gate("pinned_official_git_source", sha == PIN, sha)
    gate("official_evaluator", (root / "scripts/evaluation.py").is_file())
    gate("official_pi05_runner", (root / "src/evaluation/demo_pi05_runner.py").is_file())
    gate("official_environment", (root / "roboreel_env/src/envs").is_dir())
    gate("official_siglip_preprocessor", (root / "scripts/compute_vision_features_for_demo.py").is_file())
    gate("official_keyframe_extractor", (root / "scripts/extract_keyframes.py").is_file())
    gate("official_environment_bank",
         (root / "roboreel_env/init_states/without_distraction/push_cube.pkl").is_file())
    config = read(original) if original.is_file() else {}
    gate("official_eval_config", config.get("model") == "demo_pi05"
         and config.get("scope") == "last20"
         and config.get("max_rollout_steps") == 800
         and "chunk_size" not in config,
         "Keep upstream action_horizon=20, not Vid2Robot chunk_size=1")
    gate("checkpoint_params_directory", (ckpt / "params").is_dir() and any((ckpt / "params").iterdir()))
    train_path = next((p for p in (ckpt / "train_config.json",
                        ckpt.parent / "train_config.json") if p.is_file()), None)
    gate("checkpoint_train_config", train_path is not None)
    if train_path:
        cfg = read(train_path)
        mode = str(cfg.get("demonstration_config", "")).lower()
        gate("correct_checkpoint_sampling_variant",
             ("keyframe" in mode) == (variant == "keyframes"), mode)
        gate("trained_push_cube_task",
             "push_cube" in (cfg.get("tasks") or [cfg.get("task")]))
        cameras = cfg.get("demo_cameras") or [cfg.get("demo_camera", "front")]
        gate("front_camera", "front" in cameras, cameras)
        path_expr = str(cfg.get("data_dir", ""))
        path_expr = path_expr.replace("$" + "{ROBOREEL_DATA}", str(data))
        gate("trained_data_root_matches_eval_data",
             Path(os.path.expandvars(path_expr)).resolve() == data, cfg.get("data_dir"))
    for i in IDS:
        vid = data / f"human_demos/eval/no_distraction/push_cube/{i:04d}/front.mp4"
        feat = data / f"video_demonstration_features/{BUCKET}/push_cube/{i:04d}/front.npy"
        gate(f"human_video_{i:04d}", vid.is_file() and vid.stat().st_size > 0)
        gate(f"siglip_feature_{i:04d}", feat.is_file() and feat.stat().st_size > 0,
             "official front.npy SigLIP bundled tokens; no fabricated cache")
        if variant == "keyframes":
            annot = data / f"bpp_keyframe_annotations/{BUCKET}/push_cube/{i:04d}.json"
            valid = False
            if annot.is_file():
                try:
                    a = read(annot)
                    valid = bool(a.get("keyframe_indices")) and (
                        str(a.get("demo_id")) in (str(i), f"{i:04d}")
                    ) and a.get("task") == "push_cube" and a.get("bucket") == BUCKET
                except (ValueError, TypeError):
                    pass
            gate(f"keyframe_annotation_{i:04d}", valid)
    failures = [key for key, value in checks.items() if not value["ok"]]
    return dict(type="OFFICIAL_PI05_ND_PREFLIGHT_ONLY", variant=variant,
                method=VARIANTS[variant], pinned_source_sha=sha,
                ready_for_runtime=not failures, runtime_executed=False,
                physics_executed=False, failures=failures, checks=checks,
                checkpoint_dir=str(ckpt), data_root=str(data))


def make_config(original, seed, directory):
    if seed not in SEEDS or original.get("model") != "demo_pi05" or (
        original.get("scope") != "last20"
        or original.get("max_rollout_steps") != 800
        or "chunk_size" in original
    ):
        raise ValueError("Official pi0.5 protocol differs; abort")
    return {**original, "tasks": ["push_cube"], "seed_offset_base": seed,
            "human_demo_and_env_types":
                [["without_distraction", "without_distraction"]],
            "output_dir": str(Path(directory).resolve()), "save_video": False}


def verify(summary, seed):
    fields = dict(model="demo_pi05", runner="demo_pi05", scope="last20",
                  tasks=["push_cube"], chunk_size=20, max_rollout_steps=800,
                  seed_offset_base=seed, demo_ids_no_distract=IDS,
                  human_demo_and_env_types=
                      [["without_distraction", "without_distraction"]])
    for key, val in fields.items():
        if summary.get(key) != val:
            raise ValueError(f"Official summary incompatible: {key}")
    per = summary.get("per_trial")
    if not isinstance(per, dict) or set(per) != {CELL}:
        raise ValueError("Expected a single clean/clean ND cell")
    rows = per[CELL]
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("Expected exactly 20 physical rollouts")
    if any(not isinstance(r, dict) or r.get("task") != "push_cube"
           or type(r.get("demo_id")) is not int
           or type(r.get("success")) is not bool for r in rows):
        raise ValueError("Invalid per-trial physical record")
    if sorted(r["demo_id"] for r in rows) != IDS:
        raise ValueError("Missing/duplicate demo IDs")
    count = sum(r["success"] for r in rows)
    tab = summary.get("table", {}).get(CELL, {})
    if (tab.get("success") != count or tab.get("total") != 20
            or abs(float(tab.get("rate", -1)) - count / 20) > 1e-10):
        raise ValueError("Success table inconsistent with actual trials")
    return dict(seed_offset_base=seed, success=count, total=20,
                success_rate=count / 20, per_trial=rows)


def execute(variant, official_root, assets, out, gpu="0"):
    root, assets, out = (Path(p).resolve() for p in (official_root, assets, out))
    report = preflight(variant, root, assets)
    save(out / "preflight.json", report)
    if not report["ready_for_runtime"]:
        raise RuntimeError("Blocked by missing assets: " + ", ".join(report["failures"]))
    base = read(root / "configs/eval/nd" / (VARIANTS[variant] + ".json"))
    env = os.environ.copy()
    env.update(ROBOREEL_DATA=str(assets / "official_eval_data"),
               ROBOREEL_CKPT=str(assets / "released_checkpoints"),
               CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl",
               MUJOCO_EGL_DEVICE_ID=str(gpu))
    passes = {}
    for seed in SEEDS:
        stage = out / f"seed_offset_{seed}"
        stage.mkdir(parents=True, exist_ok=True)
        cfg = make_config(base, seed, stage)
        config_file = out / f"config_seed_offset_{seed}.json"
        if config_file.is_file() and read(config_file) != cfg:
            raise RuntimeError(f"seed={seed}: old config mismatch; do not overwrite")
        save(config_file, cfg)
        summary_file = stage / "summary.json"
        if not summary_file.is_file():
            with (stage / "run.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, "-u", "scripts/evaluation.py",
                     "--config", str(config_file), "--gpu", str(gpu)],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"seed={seed} failed: see {stage}/run.log")
        passes[seed] = verify(read(summary_file), seed)
        save(out / "partial_verified.json", {
            "status": "PARTIAL_OR_COMPLETE_VERIFIED_PASSES",
            "passes": list(passes.values()), "method": VARIANTS[variant],
            "full_3pass_complete": len(passes) == 3})
    rates = [passes[s]["success_rate"] for s in SEEDS]
    mean = sum(rates) / 3
    sem = math.sqrt(sum((x - mean) ** 2 for x in rates) / 2 / 3)
    return save(out / "official_aggregated_summary.json", {
        "type": "OFFICIAL_PI05_ND_PUSHCUBE_ACTUAL_3PASS",
        "method": VARIANTS[variant], "official_source_commit": PIN,
        "task": "push_cube", "scope": "last20", "chunk_size": 20,
        "max_rollout_steps": 800, "full_3pass_complete": True,
        "per_pass": list(passes.values()), "pooled_successes":
            sum(x["success"] for x in passes.values()), "pooled_trials": 60,
        "mean_success_rate": mean, "sem_across_seed_passes": sem,
        "only_one_official_ND_cell": True})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--official-root", default="third_party/roboreel_models")
    ap.add_argument("--assets", default="outputs/v120_baselines")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--execute", action="store_true",
                    help="Opt-in ONLY after asset gate: runs physical 20x3")
    args = ap.parse_args(argv)
    out = Path(args.out or f"outputs/v140_pi05_{args.variant}")
    if not args.execute:
        report = save(out / "preflight.json",
                      preflight(args.variant, args.official_root, args.assets))
        print("PI05_ASSET_GATE", json.dumps({
            "variant": args.variant, "ready": report["ready_for_runtime"],
            "failures": report["failures"]}), flush=True)
        return
    try:
        report = execute(args.variant, args.official_root, args.assets,
                         out, args.gpu)
    except Exception as exc:
        save(out / "status.json", {"status": "BLOCKED_OR_PARTIAL",
             "full_3pass_complete": False,
             "error": f"{type(exc).__name__}: {exc}"})
        raise
    save(out / "status.json", {"status": "ACTUAL_OFFICIAL_PI05_3PASS_VERIFIED",
         "full_3pass_complete": True})
    print("ACTUAL_PI05_PHYSICS_VERIFIED", json.dumps({
        "variant": args.variant, "k": report["pooled_successes"], "n": 60}))


if __name__ == "__main__":
    main()
