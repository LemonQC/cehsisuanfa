#!/usr/bin/env python3
"""Bounded RoboReel UniSkill DEV using the official released policy + IDM.

Uses only genuine RoboReel TRAIN human videos 0..79. It pairs two source videos
against identical StackCups initial PhysX states and torch RNG seeds. This is a
DEV smoke, not the official held-out 20x3 evaluation and not a T-RO novelty claim.
No training is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM_REL = Path("third_party/roboreel_models")
PINNED_UPSTREAM = "79a0fb0099e73c5560114753a8706276f93aefc2"
DEFAULT_DATA = Path("outputs/v120_baselines/official_eval_data")
DEFAULT_POLICY = Path("outputs/v120_baselines/released_checkpoints/nd/uniskill_policy/ckpt.pth")
DEFAULT_IDM = Path("outputs/v120_baselines/uniskill_assets/idm.pth")
DEFAULT_SKILLS = DEFAULT_DATA / "uniskill_skills"
DEFAULT_CONFIG = UPSTREAM_REL / "configs/uniskill.json"
TASKS = {"stack_cups": "StackCups-v1"}
REQUIRED_FILES = (
    "scripts/evaluation.py",
    "scripts/extract_uniskill_human_skill.py",
    "src/evaluation/uniskill_runner.py",
    "src/evaluation/utils.py",
    "src/wrappers/uniskill.py",
    "src/external_models/UniSkill/extract_skill.py",
    "src/external_models/UniSkill-Policy/robomimic/scripts/train.py",
    "src/external_models/UniSkill-Policy/configs/uniskill_policy.json",
    "roboreel_env/assets/scenes/stack_cups/dark_blue_cup/visual.glb",
    "roboreel_env/assets/scenes/stack_cups/dark_blue_cup/collision_vhacd.obj",
    "roboreel_env/assets/scenes/stack_cups/dark_blue_cup/orientation_result.json",
    "roboreel_env/assets/scenes/stack_cups/red_cup/visual.glb",
    "roboreel_env/assets/scenes/stack_cups/red_cup/collision_vhacd.obj",
    "roboreel_env/assets/scenes/stack_cups/red_cup/orientation_result.json",
    "roboreel_env/assets/scenes/stack_cups/light_blue_cup/visual.glb",
    "roboreel_env/assets/scenes/stack_cups/light_blue_cup/collision_vhacd.obj",
    "roboreel_env/assets/scenes/stack_cups/light_blue_cup/orientation_result.json",
)
REQUIRED_PACKAGES = (
    "torch", "torchvision", "transformers", "diffusers", "accelerate",
    "mani_skill", "sapien", "gymnasium", "imageio", "numpy",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], text=True,
                           capture_output=True, check=True, timeout=15)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def safe_regular(path: Path, minimum_bytes: int = 1) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size >= minimum_bytes


def validate_train_ids(ids: list[int]) -> list[int]:
    if len(ids) != 2 or len(set(ids)) != 2 or any(not isinstance(i, int) or i < 0 or i >= 80 for i in ids):
        raise ValueError("Require exactly two distinct authentic TRAIN demo IDs in [0,79]")
    return ids


def validate_skill_array(arr: np.ndarray) -> dict:
    if arr.ndim != 3 or arr.shape[1] != 1 or arr.shape[2] != 64:
        raise ValueError(f"Expected skill shape [T,1,64], got {arr.shape}")
    if arr.shape[0] < 2 or not np.isfinite(arr).all():
        raise ValueError("Skill array must have >=2 finite per-timestep vectors")
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": True}


def action_difference(a: list, b: list) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    x = np.asarray(a[:n], dtype=np.float32)
    y = np.asarray(b[:n], dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 8 or y.shape[1] != 8:
        raise ValueError("Action traces must be [T,8]")
    return float(np.linalg.norm(x - y, axis=1).mean())


def _package_versions() -> dict:
    out = {}
    for name in REQUIRED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def preflight(project: Path, data_dir: Path, policy_checkpoint: Path,
              idm_checkpoint: Path, skill_dir: Path, config_path: Path,
              task: str, source_ids: list[int]) -> dict:
    project = project.expanduser().resolve()
    source_ids = validate_train_ids(source_ids)
    if task not in TASKS:
        raise ValueError(f"Unsupported bounded DEV task: {task}")
    upstream = project / UPSTREAM_REL
    rev = git_value(upstream, "rev-parse", "HEAD") if upstream.is_dir() else None
    remote = git_value(upstream, "remote", "get-url", "origin") if upstream.is_dir() else None
    required = [{"relative_path": rel, "present": safe_regular(upstream / rel, 129)}
                for rel in REQUIRED_FILES]
    data_dir = data_dir.expanduser().resolve()
    policy_checkpoint = policy_checkpoint.expanduser().resolve()
    idm_checkpoint = idm_checkpoint.expanduser().resolve()
    skill_dir = skill_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    video_rows = []
    skill_rows = []
    for demo_id in source_ids:
        demo = f"{demo_id:04d}"
        video = data_dir / "human_demos/train/no_distraction" / task / demo / "front.mp4"
        skill = skill_dir / "without_distraction" / task / demo / "front.npy"
        video_rows.append({"id": demo_id, "present": safe_regular(video, 1024),
                           "bytes": video.stat().st_size if safe_regular(video, 1024) else 0,
                           "sha256": sha256(video) if safe_regular(video, 1024) else None})
        if safe_regular(skill, 1024):
            try:
                arr = np.load(skill, allow_pickle=False, mmap_mode="r")
                meta = validate_skill_array(arr)
                meta.update({"present": True, "sha256": sha256(skill), "bytes": skill.stat().st_size})
            except Exception as exc:
                meta = {"present": False, "error_class": type(exc).__name__}
        else:
            meta = {"present": False}
        skill_rows.append({"id": demo_id, **meta})

    # Official RoboReel UniSkillRunner has NO sibling train_config.json.
    # The released ckpt includes shape/normalization metadata, while the
    # upstream UniSkill-Policy config is pinned in REQUIRED_FILES above.

    versions = _package_versions()
    python_ok = sys.version_info[:2] == (3, 12)
    packages_ok = all(versions[k] is not None for k in REQUIRED_PACKAGES)
    cuda = {"available": False, "device_name": None}
    if versions["torch"]:
        try:
            import torch
            cuda["available"] = bool(torch.cuda.is_available())
            if cuda["available"]:
                cuda["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            cuda["error_class"] = type(exc).__name__

    base_checks = {
        "upstream_repo_present": upstream.is_dir(),
        "upstream_remote": remote,
        "upstream_revision": rev,
        "upstream_revision_pinned": rev == PINNED_UPSTREAM,
        "required_upstream_files": required,
        "policy_checkpoint_present": safe_regular(policy_checkpoint, 1024 * 1024),
        "policy_checkpoint_bytes": policy_checkpoint.stat().st_size if safe_regular(policy_checkpoint, 1024 * 1024) else 0,
        "policy_checkpoint_sha256": sha256(policy_checkpoint) if safe_regular(policy_checkpoint, 1024 * 1024) else None,
        "policy_train_config_required": False,
        "policy_architecture_config_present": safe_regular(
            upstream / "src/external_models/UniSkill-Policy/configs/uniskill_policy.json", 129),
        "idm_checkpoint_present": safe_regular(idm_checkpoint, 10 * 1024 * 1024),
        "idm_checkpoint_bytes": idm_checkpoint.stat().st_size if safe_regular(idm_checkpoint, 10 * 1024 * 1024) else 0,
        "idm_checkpoint_sha256": sha256(idm_checkpoint) if safe_regular(idm_checkpoint, 10 * 1024 * 1024) else None,
        "uniskill_config_present": safe_regular(config_path, 2),
        "train_videos": video_rows,
        "skill_arrays": skill_rows,
        "python_3_12": python_ok,
        "package_versions": versions,
        "all_required_packages_present": packages_ok,
        "cuda": cuda,
    }
    required_code_ok = bool(required) and all(x["present"] for x in required)
    base_ready = all((
        base_checks["upstream_repo_present"], base_checks["upstream_revision_pinned"],
        required_code_ok, base_checks["policy_checkpoint_present"],
        base_checks["policy_architecture_config_present"], base_checks["idm_checkpoint_present"],
        base_checks["uniskill_config_present"], all(x["present"] for x in video_rows),
        python_ok, packages_ok, cuda["available"],
    ))
    skills_ready = all(x.get("present") for x in skill_rows)
    status = "BLOCKED_BASE_ASSETS_OR_RUNTIME" if not base_ready else (
        "READY_FOR_DEV" if skills_ready else "READY_FOR_SKILL_EXTRACTION")
    return {
        "schema": "TRO1_B06_UNISKILL_STACKCUPS_PREFLIGHT_V1",
        "status": status,
        "scope": "TWO_TRAIN_VIDEOS_BOUNDED_DEV_ONLY",
        "task": task,
        "source_train_demo_ids": source_ids,
        "heldout_eval_ids_used": [],
        "new_physx_actions": 0,
        "new_training_runs": 0,
        "base_checks": base_checks,
        "skill_dir_relative_to_project": (skill_dir.relative_to(project).as_posix()
                                             if skill_dir.is_relative_to(project) else "EXTERNAL_PATH"),
    }


def save_json_once(path: Path, obj: dict) -> None:
    payload = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != payload:
            raise FileExistsError("IMMUTABLE_B06_OUTPUT_PATH_ALREADY_DIFFERENT")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(payload)


def save_progress_json(path: Path, obj: dict) -> None:
    """Atomically extend a partial trial list; never remove or reorder old rows."""
    payload = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink():
            raise FileExistsError("B06_PROGRESS_SYMLINK_REFUSED")
        old = json.loads(path.read_text(encoding="utf-8"))
        old_rows = old.get("trials", [])
        new_rows = obj.get("trials", [])
        if new_rows[:len(old_rows)] != old_rows:
            raise FileExistsError("B06_PROGRESS_MUST_BE_APPEND_ONLY")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError("B06_PROGRESS_TEMP_ALREADY_EXISTS")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


class AuditedEnv:
    def __init__(self, base):
        self.base = base
        self.n_steps = 0
        self.last_success = False
        self.action_trace = []
        self.success_trace = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, action):
        import torch
        raw = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        a = np.asarray(raw, dtype=np.float32).reshape(-1)
        if a.size != 8 or not np.isfinite(a).all():
            raise ValueError("UniSkill must produce finite 8D Panda actions")
        result = self.base.step(action)
        self.action_trace.append(a.copy())
        self.n_steps += 1
        info = result[4]
        val = info.get("success", False)
        if isinstance(val, torch.Tensor):
            val = val.reshape(-1)[0].item()
        else:
            val = bool(np.asarray(val).reshape(-1)[0]) if np.asarray(val).size else bool(val)
        self.last_success = bool(val)
        self.success_trace.append(self.last_success)
        return result


def _save_trial_video(path: Path, frames: list) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(str(path), frames, fps=20, quality=7)
        return None
    except Exception as exc:
        return type(exc).__name__


def run_dev(plan: dict, project: Path, data_dir: Path, policy_checkpoint: Path,
            idm_checkpoint: Path, skill_dir: Path, config_path: Path,
            out: Path, episodes: int, max_steps: int, save_video: bool,
            publish: bool) -> dict:
    if plan["status"] != "READY_FOR_DEV":
        raise RuntimeError("B06_DEV_BLOCKED: " + plan["status"])
    if not 1 <= episodes <= 2:
        raise ValueError("Bounded B06 DEV permits 1..2 paired episodes (2..4 rollouts)")
    if not 1 <= max_steps <= 800:
        raise ValueError("Bounded B06 DEV permits 1..800 steps per rollout")
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("B06_OUTPUT_ALREADY_EXISTS_NO_OVERWRITE: " + str(out))

    project = project.resolve()
    upstream = project / UPSTREAM_REL
    data_dir = data_dir.resolve()
    policy_checkpoint = policy_checkpoint.resolve()
    idm_checkpoint = idm_checkpoint.resolve()
    skill_dir = skill_dir.resolve()
    config_path = config_path.resolve()

    # Pin the headless render path before simulator imports.
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    for p in (upstream, upstream / "src", upstream / "src/external_models",
              upstream / "src/external_models/UniSkill",
              upstream / "src/external_models/UniSkill-Policy",
              upstream / "roboreel_env/src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    torch.set_num_threads(4)
    from evaluation.uniskill_runner import UniSkillRunner
    from evaluation.utils import make_eval_env

    runner = UniSkillRunner(
        policy_checkpoint=str(policy_checkpoint),
        skill_dir=str(skill_dir),
        uniskill_config=str(config_path),
        data_dir=str(data_dir),
        tasks=[plan["task"]],
        demo_camera="front",
        robot_cameras=["base_camera"],
        idm_checkpoint=str(idm_checkpoint),
    )
    source_ids = plan["source_train_demo_ids"]
    source_skills = {i: runner.load_demo_static(False, plan["task"], i) for i in source_ids}
    for i, skill in source_skills.items():
        validate_skill_array(np.asarray(skill))

    # Verify original StackCups meshes; this task does not use PartNet-Mobility.
    scene_root = upstream / "roboreel_env/assets/scenes/stack_cups"
    for cup in ("dark_blue_cup", "red_cup", "light_blue_cup"):
        for asset in ("visual.glb", "collision_vhacd.obj", "orientation_result.json"):
            if not safe_regular(scene_root / cup / asset, 129):
                raise RuntimeError("PINNED_ROBOREEL_STACK_CUPS_SCENE_ASSET_MISSING")

    out.mkdir(parents=True, exist_ok=False)
    save_json_once(out / "preflight.json", plan)
    manifest = {
        "schema": "TRO1_B06_UNISKILL_STACKCUPS_DEV_V1",
        "status": "RUNNING_BOUNDED_DEV_NOT_FORMAL",
        "task": plan["task"],
        "source_train_demo_ids": source_ids,
        "episodes": episodes,
        "max_steps": max_steps,
        "official_upstream_revision": plan["base_checks"]["upstream_revision"],
        "policy_checkpoint_sha256": plan["base_checks"]["policy_checkpoint_sha256"],
        "idm_checkpoint_sha256": plan["base_checks"]["idm_checkpoint_sha256"],
        "source_video_sha256": {str(x["id"]): x["sha256"] for x in plan["base_checks"]["train_videos"]},
        "skill_sha256": {str(x["id"]): x["sha256"] for x in plan["base_checks"]["skill_arrays"]},
        "same_reset_state_within_pair": True,
        "same_torch_rng_seed_within_pair": True,
        "train_videos_only": True,
        "no_heldout_eval_videos": True,
        "no_policy_training": True,
        "no_partnet_dependency": True,
        "output_dir": str(out),
    }
    save_json_once(out / "manifest.json", manifest)

    trials = []
    for episode in range(episodes):
        seed = 310000 + episode
        torch_seed = 560000 + episode
        env, _ = make_eval_env(plan["task"], False, max_episode_steps=max_steps)
        try:
            env.reset(seed=seed)
            snapshot = copy.deepcopy(env.unwrapped.get_state_dict())
            order = source_ids if episode % 2 == 0 else list(reversed(source_ids))
            for order_idx, source_id in enumerate(order):
                env.reset(seed=seed)
                env.unwrapped.set_state_dict(copy.deepcopy(snapshot))
                obs = env.unwrapped.get_obs()
                torch.manual_seed(torch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(torch_seed)
                frames = [] if save_video else None
                logged = AuditedEnv(env)
                sub = out / f"episode_{episode:02d}" / f"source_{source_id:04d}"
                sub.mkdir(parents=True, exist_ok=False)
                t0 = time.monotonic()
                success = False
                error_class = None
                error_message = None
                try:
                    success = bool(runner.rollout(logged, obs, source_skills[source_id],
                                                  max_steps, 1, frames))
                except Exception as exc:
                    error_class = type(exc).__name__
                    error_message = str(exc)[:700]
                action_path = sub / "actions.npy"
                np.save(action_path, np.asarray(logged.action_trace, dtype=np.float32).reshape(-1, 8))
                save_json_once(sub / "success_trace.json", {"success": logged.success_trace})
                video_error = _save_trial_video(sub / "rollout.mp4", frames) if frames else None
                video_path = sub / "rollout.mp4"
                trial = {
                    "episode": episode,
                    "reset_seed": seed,
                    "torch_rng_seed": torch_seed,
                    "condition_order": order_idx,
                    "source_train_demo_id": source_id,
                    "success": bool(success and logged.last_success),
                    "runner_return_success": bool(success),
                    "last_success": logged.last_success,
                    "steps": logged.n_steps,
                    "error_class": error_class,
                    "error_message": error_message,
                    "video_error_class": video_error,
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "actions_sha256": sha256(action_path),
                    "video_sha256": sha256(video_path) if safe_regular(video_path, 1024) else None,
                    "no_privileged_state_as_policy_input": True,
                }
                save_json_once(sub / "status.json", trial)
                trials.append(trial)
                save_progress_json(out / "per_trial.partial.json", {"trials": trials})
                print("ROBOREEL_B06_UNISKILL_TRIAL", json.dumps({k: trial[k] for k in
                      ("episode", "source_train_demo_id", "success", "steps", "error_class")}), flush=True)
                if error_class:
                    raise RuntimeError("B06_REAL_ROLLOUT_ERROR_STOP: " + error_class + ": " + str(error_message))
        finally:
            env.close()

    pairs = []
    for episode in range(episodes):
        p = [x for x in trials if x["episode"] == episode]
        if len(p) != 2:
            raise RuntimeError("B06_INCOMPLETE_PAIRED_EPISODE")
        action_paths = [out / f"episode_{episode:02d}" / f"source_{x['source_train_demo_id']:04d}" / "actions.npy" for x in p]
        a, b = [np.load(x).tolist() for x in action_paths]
        pairs.append({
            "episode": episode,
            "same_reset_seed": p[0]["reset_seed"] == p[1]["reset_seed"],
            "same_torch_rng_seed": p[0]["torch_rng_seed"] == p[1]["torch_rng_seed"],
            "source_ids": [x["source_train_demo_id"] for x in p],
            "successes": [x["success"] for x in p],
            "success_disagree": p[0]["success"] != p[1]["success"],
            "mean_action_l2_first_common_steps": action_difference(a[:20], b[:20]),
        })
    result = {
        "schema": "TRO1_B06_UNISKILL_STACKCUPS_DEV_V1",
        "status": "DEV_COMPLETE_NOT_FORMAL",
        "task": plan["task"],
        "episodes": episodes,
        "rollouts": len(trials),
        "successes": sum(x["success"] for x in trials),
        "trials": trials,
        "pairs": pairs,
        "any_pair_success_difference": any(x["success_disagree"] for x in pairs),
        "new_training_runs": 0,
        "new_physx_actions": sum(x["steps"] for x in trials),
        "heldout_eval_videos_used": 0,
        "paper_ready": False,
    }
    save_json_once(out / "results.json", result)
    summary = [
        "# B06 UniSkill on RoboReel StackCups — bounded DEV",
        "",
        f"- Official UniSkill policy checkpoint; official ISD/IDM checkpoint; upstream `{manifest['official_upstream_revision']}`.",
        f"- Genuine TRAIN videos only: {source_ids}; no held-out EVAL videos.",
        f"- Real PhysX rollouts: {len(trials)}; success {result['successes']}/{len(trials)}.",
        f"- Paired episodes with success disagreement: {sum(x['success_disagree'] for x in pairs)}/{episodes}.",
        "- Same reset state and Torch RNG seed within each pair; source order swapped across episodes.",
        "- No training, no PartNet dependency, no privileged state as policy input.",
        "- DEV only: not the official last20 × 3 held-out evaluation and not evidence of T-RO novelty.",
        "- See `results.json`, `per_trial.csv`, per-rollout action traces and videos.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    with (out / "per_trial.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reset_seed", "torch_rng_seed", "source_train_demo_id",
                                         "success", "steps", "error_class", "elapsed_seconds"])
        w.writeheader()
        for t in trials:
            w.writerow({k: t[k] for k in w.fieldnames})
    print("ROBOREEL_B06_UNISKILL_RESULT", json.dumps({"status": result["status"],
          "successes": f"{result['successes']}/{len(trials)}", "physx_actions": result["new_physx_actions"],
          "summary": str(out / "SUMMARY.md")}), flush=True)

    if publish:
        publish_existing(out)
    return result



def publish_existing(out: Path) -> str:
    """Publish a *finished* prior B06 result without importing torch/PhysX.

    Useful after a network interruption between the last rollout and the
    GitHub upload. All local SHA256 hashes are reverified, so this does not
    accidentally report a modified/partial experiment as complete.
    """
    out = Path(out).expanduser().resolve()
    allowed = PROJECT / "outputs/research_runs"
    if not out.is_relative_to(allowed.resolve()):
        raise ValueError("B06_PUBLISH_PATH_OUTSIDE_PROJECT_OUTPUTS")
    mf, rf = out / "manifest.json", out / "results.json"
    if not safe_regular(mf, 64) or not safe_regular(rf, 64):
        raise FileNotFoundError("B06_COMPLETE_MANIFEST_AND_RESULTS_REQUIRED")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    result = json.loads(rf.read_text(encoding="utf-8"))
    schema = "TRO1_B06_UNISKILL_STACKCUPS_DEV_V1"
    if (result.get("schema") != schema or manifest.get("schema") != schema
        or result.get("status") != "DEV_COMPLETE_NOT_FORMAL"
        or manifest.get("status") != "RUNNING_BOUNDED_DEV_NOT_FORMAL"
        or result.get("task") != "stack_cups"
        or result.get("heldout_eval_videos_used") != 0
        or result.get("new_training_runs") != 0):
        raise ValueError("B06_NOT_A_VALID_COMPLETED_DEV_RESULT")
    trials = result.get("trials")
    n = result.get("episodes")
    if (not isinstance(trials, list) or n not in (1, 2)
        or len(trials) != 2*n or result.get("rollouts") != 2*n
        or len(result.get("pairs", [])) != n):
        raise ValueError("B06_TRIAL_COUNT_OR_PAIRS_INVALID")
    seen = set()
    for t in trials:
        ep, sid = t["episode"], t["source_train_demo_id"]
        if (not isinstance(ep, int) or not 0 <= ep < n
            or not isinstance(sid, int) or sid not in manifest["source_train_demo_ids"]
            or (ep,sid) in seen):
            raise ValueError("B06_TRIAL_ID_INVALID_OR_DUPLICATE")
        seen.add((ep,sid))
        folder = out / f"episode_{ep:02d}" / f"source_{sid:04d}"
        actions = folder / "actions.npy"
        if not safe_regular(actions, 1) or sha256(actions) != t.get("actions_sha256"):
            raise ValueError("B06_ACTION_TRACE_HASH_MISMATCH")
        video_hash = t.get("video_sha256")
        if video_hash:
            video = folder / "rollout.mp4"
            if not safe_regular(video, 1024) or sha256(video) != video_hash:
                raise ValueError("B06_VIDEO_HASH_MISMATCH")
    if result.get("successes") != sum(bool(t["success"]) for t in trials):
        raise ValueError("B06_SUCCESS_COUNT_MISMATCH")
    public = {
        "schema": schema,
        "status": result["status"],
        "task": result["task"],
        "episodes": n,
        "rollouts": len(trials),
        "successes": result["successes"],
        "new_physx_actions": result["new_physx_actions"],
        "new_training_runs": 0,
        "official_upstream_revision": manifest["official_upstream_revision"],
        "policy_checkpoint_sha256": manifest["policy_checkpoint_sha256"],
        "idm_checkpoint_sha256": manifest["idm_checkpoint_sha256"],
        "source_video_sha256": manifest["source_video_sha256"],
        "skill_sha256": manifest["skill_sha256"],
        "trials": [{k: t[k] for k in ("episode", "reset_seed", "torch_rng_seed",
                                        "source_train_demo_id", "success", "steps",
                                        "error_class", "actions_sha256", "video_sha256")}
                   for t in trials],
        "pairs": result["pairs"],
        "heldout_eval_videos_used": 0,
        "formal_evaluation": False,
    }
    raw = (json.dumps(public, indent=2, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    uid = hashlib.sha256(raw).hexdigest()[:16]
    from scripts.publish_research_evidence import assert_evidence_branch_exists, write_immutable_report
    assert_evidence_branch_exists()
    remote_path = f"evidence/runs/b06_uniskill_stackcups_dev_{uid}/evidence.json"
    published = write_immutable_report(remote_path, raw)
    url = f"https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/{remote_path}"
    print("ROBOREEL_B06_UNISKILL_PUBLISHED", json.dumps({"result": published, "url": url}), flush=True)
    return url


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--data-dir", type=Path, default=PROJECT / DEFAULT_DATA)
    p.add_argument("--policy-checkpoint", type=Path, default=PROJECT / DEFAULT_POLICY)
    p.add_argument("--idm-checkpoint", type=Path, default=Path(os.environ.get("UNISKILL_ISD", PROJECT / DEFAULT_IDM)))
    p.add_argument("--skill-dir", type=Path, default=PROJECT / DEFAULT_SKILLS)
    p.add_argument("--config", type=Path, default=PROJECT / DEFAULT_CONFIG)
    p.add_argument("--task", choices=sorted(TASKS), default="stack_cups")
    p.add_argument("--source-ids", type=int, nargs=2, default=[0, 1])
    p.add_argument("--out", type=Path, default=PROJECT / "outputs/research_runs/b06_uniskill_stackcups_dev_v1")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--gpu", default="0")
    p.add_argument("--preflight", action="store_true", help="Read-only local/runtime check; no PhysX")
    p.add_argument("--run", action="store_true", help="Explicitly run 2..4 genuine PhysX DEV rollouts")
    p.add_argument("--publish", action="store_true", help="Publish sanitized result to evidence branch")
    p.add_argument("--publish-existing", action="store_true", help="Re-publish an existing, verified complete DEV result with NO PhysX")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args(argv)
    if not args.preflight and not args.run and not args.publish_existing:
        p.error("choose --preflight, --run or --publish-existing")
    if args.publish and not args.run:
        p.error("--publish requires --run")
    if args.publish_existing:
        if args.run or args.preflight:
            p.error("--publish-existing cannot be combined with --run/--preflight")
        publish_existing(args.out)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    plan = preflight(args.project, args.data_dir, args.policy_checkpoint,
                     args.idm_checkpoint, args.skill_dir, args.config,
                     args.task, args.source_ids)
    print("ROBOREEL_B06_UNISKILL_PREFLIGHT", json.dumps(plan, ensure_ascii=False), flush=True)
    if args.preflight:
        return 0 if plan["status"] != "BLOCKED_BASE_ASSETS_OR_RUNTIME" else 2
    run_dev(plan, args.project, args.data_dir, args.policy_checkpoint,
            args.idm_checkpoint, args.skill_dir, args.config, args.out,
            args.episodes, args.max_steps, not args.no_video, args.publish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
