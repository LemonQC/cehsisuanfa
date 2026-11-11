#!/usr/bin/env python3
"""M1-B3 strict offline verifier: actual four-case PhysX outcomes only.

Geometric preflight, single-run direct-push failure and clear-first
script success are all INSUFFICIENT to prove causal necessity or novelty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

from agcofl.m1b3_corridor_physics import (
    MODES, SEED, GOAL_RADIUS, BLOCKER_X, PARKING_Y,
    geometry_preflight,
)


def xyz(row, key):
    v=row.get(key)
    if (not isinstance(v,list) or len(v)!=3
            or any(type(z) not in (int,float) or not math.isfinite(z)
                   for z in v)):
        raise ValueError(f"Malformed finite 3D measured {key}")
    return tuple(v)


def length(a,b):
    return math.dist(a,b)


def summarize(folder: Path):
    geometry=geometry_preflight()
    if not geometry["pass"]:
        raise ValueError("Preflight geometry rejected")
    rows={}
    missing=[]
    for mode in MODES:
        file=folder/mode/"status.json"
        if not file.is_file():
            missing.append(mode)
            continue
        obj=json.loads(file.read_text())
        if obj.get("status")=="M1B3_EXECUTION_ERROR_INVALID_PHYSICS":
            return {
                "type":"M1B3_PHYSICS_COUNTERFACTUAL_GATE",
                "status":"PHYSICS_CASE_ERROR_NOT_EVIDENCE",
                "mode":mode,"error":obj.get("error"),
                "geometry_preflight_pass":True,
                "pilot_contrast_pass":False,
                "causal_necessity_proven":False,
                "strong_tamp_evaluated":False,
                "paper_ready":False,
            }
        if (obj.get("kind") not in ("source_pick_to_same_goal",mode)
                or obj.get("mode")!=mode
                or obj.get("same_goal_scene_version")!="m1b3_corridor_v0"
                or obj.get("scene")!="AGCOFLM1B3Corridor-v0"
                or int(obj.get("seed",-1))!=SEED):
            raise ValueError(f"{mode}: not the same frozen M1B3 scene/seed")
        if not isinstance(obj.get("gate"),dict) or type(obj["gate"].get("valid")) is not bool:
            raise ValueError(f"{mode}: missing measured valid flag")
        if type(obj.get("steps")) is not int or obj["steps"]<=0:
            raise ValueError(f"{mode}: missing actual PhysX actions")
        xyz(obj,"state_before")
        xyz(obj,"goal")
        xyz(obj,"final")
        xyz(obj,"blocker_before")
        rows[mode]=obj

    if missing:
        return dict(type="M1B3_PHYSICS_COUNTERFACTUAL_GATE",
                    status="INCOMPLETE_REAL_PHYSICS_CASES",
                    missing=missing,geometry_preflight_pass=True,
                    pilot_contrast_pass=False,
                    causal_necessity_proven=False,
                    strong_tamp_evaluated=False,paper_ready=False)
    source=rows["blocked_source"]
    start=xyz(source,"state_before")
    goal=xyz(source,"goal")
    if abs(goal[0]-.210)>1e-3 or abs(goal[1])>1e-3:
        raise ValueError("M1B3 final goal differs from prespecified task")
    for mode,obj in rows.items():
        if length(start,xyz(obj,"state_before"))>1e-4:
            raise ValueError(f"{mode}: initial task object differs")
        if length(goal,xyz(obj,"goal"))>1e-4:
            raise ValueError(f"{mode}: final task goal differs")
        blocker=xyz(obj,"blocker_before")
        expected_y=PARKING_Y if mode=="open_direct" else 0.0
        if abs(blocker[0]-BLOCKER_X)>1e-4 or abs(blocker[1]-expected_y)>1e-4:
            raise ValueError(f"{mode}: blocker state does not match counterfactual")
    scores={name:row["gate"]["valid"] for name,row in rows.items()}
    direct_block_moved=rows["blocked_direct"]["gate"].get("blocker_moved_m")
    direct_block_cleared=rows["blocked_direct"]["gate"].get(
        "blocker_cleared_to_side_pocket")
    clear_blocker=bool(rows["blocked_clear"]["gate"].get(
        "blocker_cleared_to_side_pocket",False))
    # Do not mislabel a controller failure as proof all other target
    # plans must fail. This is only the controlled scripted contrast.
    empirical_contrast=(
        scores["blocked_source"]
        and not scores["blocked_direct"]
        and scores["blocked_clear"]
        and scores["open_direct"]
        and clear_blocker
    )
    return {
        "type":"M1B3_PHYSICS_COUNTERFACTUAL_GATE",
        "status":("PHYSICS_SCRIPTED_CONTRAST_OBSERVED_NEEDS_STRONG_TAMP"
                  if empirical_contrast
                  else "PHYSICS_CONTRAST_INCONCLUSIVE_OR_FAILED"),
        "geometry_preflight_pass":True,
        "matched_task_start_and_goal":True,
        "source_pick_place_success":scores["blocked_source"],
        "blocked_direct_push_success":scores["blocked_direct"],
        "blocked_clear_then_push_success":scores["blocked_clear"],
        "open_direct_push_success":scores["open_direct"],
        "blocked_direct_observed_blocker_displacement_m":direct_block_moved,
        "blocked_direct_observed_side_clearance":direct_block_cleared,
        "clear_side_pocket_observed":clear_blocker,
        "pilot_contrast_pass":bool(empirical_contrast),
        "causal_necessity_proven":False,
        "strong_tamp_evaluated":False,
        "learned_method_evaluated":False,
        "source_video_information_value_measured":False,
        "paper_ready":False,
        "caution":"Stationary-blocker C-space and scripted direct-vs-clear controls do not exhaust dynamic alternatives, learned baselines, or an equal-info TAMP.",
        "per_case":{name:{
            "success":row["gate"]["valid"],"steps":row["steps"],
            "goal_error_m":row["gate"].get("m1b3_goal_error_m",
                                          row["gate"].get("goal_error_m")),
            "error":row.get("error"),
            "artifact_error":row.get("artifact_error"),
        } for name,row in rows.items()},
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,
                   default=Path("outputs/m1b3_corridor_dev"))
    args=p.parse_args(argv)
    root=args.out.resolve()
    report=summarize(root)
    destination=root/"counterfactual_gate.json"
    if destination.is_file() and json.loads(destination.read_text())!=report:
        raise FileExistsError(
            f"Existing frozen gate differs; inspect raw outputs: {destination}")
    root.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print("M1B3_PHYSICS_GATE",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
