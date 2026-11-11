#!/usr/bin/env python3
"""Offline diagnosis of existing M1-B3 PhysX JSON traces. NO simulator, GPU or training.

Reports raw SHA256, true per-phase object motion and source MPLib failures.
Never rewrites the original four trials or treats an unsuccessful oracle
as evidence that a different policy or TAMP cannot succeed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODES = ("blocked_source", "blocked_direct", "blocked_clear", "open_direct")
SCENE = "AGCOFLM1B3Corridor-v0"


def xyz(v, name):
    if (not isinstance(v, list) or len(v) != 3 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in v)):
        raise ValueError(f"{name}: not 3 finite coordinates")
    return [float(x) for x in v]


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hash_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_case(root, mode):
    status_file = root / mode / "status.json"
    trace_file = root / mode / "physical_trace.json"
    if not status_file.is_file() or not trace_file.is_file():
        raise FileNotFoundError(f"Missing original M1B3 saved files: {mode}")
    row = json.loads(status_file.read_text())
    trace = json.loads(trace_file.read_text())
    if (row.get("scene") != SCENE or row.get("mode") != mode or
            row.get("seed") != 120 or
            row.get("same_goal_scene_version") != "m1b3_corridor_v0"):
        raise ValueError(f"{mode}: wrong original scene or seed")
    if row.get("status") == "M1B3_EXECUTION_ERROR_INVALID_PHYSICS":
        raise ValueError(f"{mode}: saved simulator execution error, not a valid measured trial")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{mode}: no actual actions in physical trace")
    if row.get("steps") != len(trace):
        raise ValueError(f"{mode}: stored action count disagrees with trace length")
    for key in ("state_before", "goal", "final", "blocker_before"):
        xyz(row.get(key), f"{mode} {key}")
    for index, sample in enumerate(trace):
        if not isinstance(sample, dict) or sample.get("t") != index:
            raise ValueError(f"{mode}: invalid physical trace index {index}")
        for key in ("task_cube", "blocker", "tcp"):
            xyz(sample.get(key), f"{mode} trace[{index}].{key}")
    hashes = {
        "status_sha256": hash_file(status_file),
        "trace_sha256": hash_file(trace_file),
        "status_path": str(status_file),
        "trace_path": str(trace_file),
    }
    return row, trace, hashes


def metrics(trace, initial_cube, initial_blocker, goal):
    task = [initial_cube] + [x["task_cube"] for x in trace]
    blocker = [initial_blocker] + [x["blocker"] for x in trace]
    tcp = [x["tcp"] for x in trace]
    total = lambda points: sum(dist2(a, b) for a, b in zip(points, points[1:]))
    tail = [x["task_cube"] for x in trace[-15:]]
    return {
        "steps": len(trace),
        "task_start_xy_m": task[0][:2],
        "task_final_xy_m": task[-1][:2],
        "task_net_xy_m": dist2(task[0], task[-1]),
        "task_total_xy_m": total(task),
        "task_last_15_steps_xy_m": total(tail),
        "task_closest_goal_error_m": min(dist2(p, goal) for p in task),
        "task_final_goal_error_m": dist2(task[-1], goal),
        "task_max_z_m": max(p[2] for p in task),
        "blocker_start_xy_m": blocker[0][:2],
        "blocker_final_xy_m": blocker[-1][:2],
        "blocker_net_xy_m": dist2(blocker[0], blocker[-1]),
        "blocker_max_z_m": max(p[2] for p in blocker),
        "tcp_total_xy_m": total(tcp),
        "task_grasped_steps": sum(bool(x.get("is_grasping_task")) for x in trace),
        "sim_success_logged_steps": sum(bool(x.get("env_success")) for x in trace),
        "task_near_throat_steps": sum(abs(x["task_cube"][0] - 0.158) < .022 for x in trace),
    }


def analyze(root):
    rows, traces, sha = {}, {}, {}
    for mode in MODES:
        rows[mode], traces[mode], sha[mode] = read_case(root, mode)
    source_start = rows["blocked_source"]["state_before"]
    source_goal = rows["blocked_source"]["goal"]
    for mode in MODES:
        if (dist2(rows[mode]["state_before"], source_start) > 1e-4 or
                dist2(rows[mode]["goal"], source_goal) > 1e-4):
            raise ValueError(f"{mode}: not the same physical cube start and goal")
    cases = {}
    for mode in MODES:
        row, trace = rows[mode], traces[mode]
        entry = {
            "success": row.get("gate", {}).get("valid"),
            "steps": row["steps"],
            "error": row.get("error"),
            "artifact_error": row.get("artifact_error"),
            "gate": row.get("gate", {}),
            "measured": metrics(trace, row["state_before"], row["blocker_before"], row["goal"]),
            "original_files": sha[mode],
        }
        if mode == "blocked_source":
            entry["official_planner_phases"] = row.get("planner_phases", [])
        else:
            phases = row.get("phases", [])
            if not isinstance(phases, list):
                raise ValueError(f"{mode}: no phase list")
            cursor = 0
            entry["physical_phase_segments"] = []
            for phase in phases:
                n = phase.get("steps")
                if type(n) is not int or n < 0 or cursor+n > len(trace):
                    raise ValueError(f"{mode}: phase counts do not match trace")
                segment = trace[cursor:cursor+n]
                local = (metrics(segment,
                    row["state_before"] if cursor == 0 else trace[cursor-1]["task_cube"],
                    row["blocker_before"] if cursor == 0 else trace[cursor-1]["blocker"],
                    row["goal"]) if segment else {"steps": 0})
                entry["physical_phase_segments"].append({
                    "phase": phase.get("name"),
                    "phase_success": phase.get("success"),
                    "phase_failure": phase.get("failure"),
                    "phase_budget_exhausted": phase.get("exhausted"),
                    "distance_to_subgoal_m": phase.get("distance_to_subgoal_m"),
                    "measured": local,
                })
                cursor += n
            if cursor != len(trace):
                raise ValueError(f"{mode}: phase lengths do not sum to actual steps")
        cases[mode] = entry

    conclusions = []
    if cases["blocked_source"]["error"]:
        conclusions.append("Source planning error logged; inspect phases and cube pose at lower failure. No feasibility conclusion follows.")
    if (cases["blocked_clear"]["gate"].get("blocker_cleared_to_side_pocket") and
            not cases["blocked_clear"]["success"]):
        conclusions.append("Blocker side clearance observed, but cube task failed. Inspect the two measured phases separately before changing geometry or controller.")
    if cases["open_direct"]["success"] and not cases["blocked_direct"]["success"]:
        conclusions.append("Open vs blocked is an observed scripted-push contrast, not proof that all alternate dynamic Push plans fail.")
    return {
        "type": "M1B3_EXISTING_PHYSICS_READ_ONLY_DIAGNOSIS",
        "scene": SCENE, "seed": 120,
        "same_start_and_goal": True,
        "cases": cases, "interpretation_limits": conclusions,
        "ran_simulator": False, "modified_controller": False,
        "causal_necessity_proven": False,
        "next_decision": "Use measured failed phase, contact and relative TCP/object motion to justify at most one explicitly versioned change; freeze original four raw episodes.",
    }


def markdown_report(report):
    lines = ["# M1-B3 saved PhysX trace diagnosis", "",
             "No PhysX episode, model training or parameter tuning was executed.", "",
             "| Case | Valid | Steps | Measured final XY error | Failure / phases |",
             "|---|---|---:|---:|---|"]
    for mode in MODES:
        case = report["cases"][mode]
        phase_desc = ("; ".join(f"{p['phase']}: {p['phase_success']} "
                    f"({p['measured']['steps']} steps)"
                    for p in case.get("physical_phase_segments", [])))
        lines.append(f"| {mode} | {case['success']} | {case['steps']} | "
                     f"{case['measured']['task_final_goal_error_m']:.4f} m | "
                     f"{case['error'] or phase_desc or 'see original phases'} |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend("- " + x for x in report["interpretation_limits"])
    lines.extend(["", report["next_decision"], "",
                  "Source SHA256 values and full numeric phase breakdown: diagnosis.json.",
                  "This is not a causal-necessity proof or a strong TAMP result.", ""])
    return "\n".join(lines)


def save_once(path, content):
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to overwrite original diagnosis: {path}")
        return
    path.write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("outputs/m1b3_corridor_dev"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = analyze(root)
    save_once(root / "diagnosis.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    save_once(root / "DIAGNOSIS.md", markdown_report(report))
    print("M1B3_OFFLINE_DIAG", json.dumps({
        "ran_simulator": False,
        "source_error": report["cases"]["blocked_source"]["error"],
        "source_planner_phases": report["cases"]["blocked_source"].get("official_planner_phases"),
        "clear_phase_metrics": report["cases"]["blocked_clear"].get("physical_phase_segments"),
        "blocked_direct_measured": report["cases"]["blocked_direct"]["measured"],
        "open_direct_success": report["cases"]["open_direct"]["success"],
        "report": str(root / "diagnosis.json"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
