"""Live observability for the ORIGINAL RoboReel released policy evaluator.

Attaches a *transparent* instrumentation wrapper to the authors' runner
instance. The original functions, inputs, return values, action chunking,
checkpoint, simulator, success criteria and per-trial official JSON remain
UNCHANGED. Never fabricate completed episodes from elapsed time.

A process already running before this file is pulled cannot gain these logs:
the new wrapper takes effect on subsequent invocations only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def install_official_progress(runner, output_dir, seed_offset: int,
                              total: int = 20):
    """Attach progress printouts and an atomically updated progress.json.

    Official run_eval_protocol calls load_demo_static then exactly one rollout
    for each real RoboReel (task, demo_id). This wrapper preserves both method
    return values, executes all original side effects exactly once, and
    records a result only AFTER the original rollout actually returns.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "progress.json"
    events_path = out / "progress.jsonl"
    original_load = runner.load_demo_static
    original_rollout = runner.rollout
    count = 0
    successes = 0
    active = None
    started = time.monotonic()
    # The official evaluator itself is the sole scientific source of final
    # 20-row results; this live progress file is informative, NOT a paper score.
    state = {
        "type": "UNOFFICIAL_RUNNER_PROGRESS_TELEMETRY_ONLY",
        "status": "RUNNING",
        "seed_offset_base": int(seed_offset),
        "total_expected": int(total),
        "completed": 0,
        "successes_so_far": 0,
        "percentage": 0.0,
        "active_demo_id": None,
        "last_completed": None,
        "authoritative_results_path": str(out / "summary.json"),
        "note": ("Only original official run_eval_protocol summary.json "
                 "can validate completed 20-demo paper results."),
    }
    # Clear only stale progress TELEMETRY when a fresh evaluator starts.
    events_path.write_text("")

    def record(kind, info):
        payload = {
            "event": kind, "utc": datetime.now(timezone.utc).isoformat(),
            "seed_offset_base": int(seed_offset), **info
        }
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        state["last_update_utc"] = payload["utc"]
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        os.replace(tmp, status_path)
        print("OFFICIAL_V130_TRIAL_" + kind, json.dumps(payload), flush=True)

    record("MONITOR_START", {"total": total})

    def load_wrapped(demo_distract, task, demo_id):
        nonlocal active
        active = {
            "demo_distract": bool(demo_distract),
            "task": str(task), "demo_id": int(demo_id),
            "index": count + 1,
            "started": time.monotonic(),
        }
        state["active_demo_id"] = int(demo_id)
        state["status"] = "RUNNING"
        record("START", {
            "index": count + 1, "total": total, "task": str(task),
            "demo_id": int(demo_id),
            "phase": "load_demo_and_prepare_simulator",
        })
        try:
            return original_load(demo_distract, task, demo_id)
        except BaseException as error:
            state["status"] = "ERROR"
            record("ERROR", {
                "index": count + 1, "demo_id": int(demo_id),
                "stage": "original_load_demo_static",
                "error_type": type(error).__name__,
            })
            raise

    def rollout_wrapped(env, obs, demo_static, max_steps, chunk_size, video_frames):
        nonlocal count, successes, active
        if active is None:
            raise RuntimeError("Original rollout called without demo source metadata")
        this = active
        try:
            result = original_rollout(
                env, obs, demo_static, max_steps, chunk_size, video_frames)
        except BaseException as error:
            state["status"] = "ERROR"
            record("ERROR", {
                "index": this["index"], "demo_id": this["demo_id"],
                "stage": "original_rollout",
                "error_type": type(error).__name__,
            })
            raise
        # Only a RETURNED original physical success flag counts.
        success = bool(result)
        count += 1
        successes += int(success)
        state.update(
            completed=count,
            successes_so_far=successes,
            percentage=round(100 * count / total, 2),
            active_demo_id=None,
            last_completed={
                "task": this["task"], "demo_id": this["demo_id"],
                "success": success,
                "duration_s": round(time.monotonic() - this["started"], 2),
            },
            status="RUNNING" if count < total else "ALL_TRIALS_RETURNED_AWAIT_OFFICIAL_SUMMARY",
        )
        record("DONE", {
            "index": count, "total": total,
            "demo_id": this["demo_id"], "success": success,
            "successes_so_far": successes,
            "percentage": state["percentage"],
            "trial_elapsed_s": state["last_completed"]["duration_s"],
            "pass_elapsed_s": round(time.monotonic() - started, 1),
        })
        active = None
        return result

    runner.load_demo_static = load_wrapped
    runner.rollout = rollout_wrapped
    return status_path
