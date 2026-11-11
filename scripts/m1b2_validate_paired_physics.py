#!/usr/bin/env python3
"""Read ONLY the three M1-B2 PhysX statuses; no simulator, no checkpoint.

Never turns a valid nominal script or 2D path into a physics claim.
Requires exact frozen source/target initial state and object goal.
M1-B2 is a *feasibility gate*, not a learned capability method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math


MODES=("source","target_direct","target_clear")


def _coords(obj,key):
    v=obj.get(key)
    if (not isinstance(v,list) or len(v)!=3
            or any(not isinstance(x,(float,int)) or not math.isfinite(x)
                   for x in v)):
        raise ValueError(f"Missing/malformed physical vector: {key}")
    return v


def dist(a,b):
    return sum((float(x)-float(y))**2 for x,y in zip(a,b))**0.5


def summarize(root):
    data={}
    missing=[]
    for mode in MODES:
        file=root/mode/"status.json"
        if not file.is_file():
            missing.append(mode)
            continue
        row=json.loads(file.read_text())
        if row.get("status")=="M1B2_EXECUTION_ERROR_NOT_VALID_EVIDENCE":
            return {
                "type":"M1B2_SAME_GOAL_PHYSX_GATE",
                "status":"CASE_EXECUTION_ERROR_NO_PAIRED_PHYSICS_CLAIM",
                "failed_mode":mode,
                "error":row.get("error"),
                "passed":False,
                "paper_ready":False,
            }
        if row.get("kind")!=(
                "source_pick_to_same_goal" if mode=="source" else mode):
            raise ValueError(f"Not a valid measured {mode} file: {file}")
        data[mode]=row
    if missing:
        return dict(
            type="M1B2_SAME_GOAL_PHYSX_GATE",
            status="INCOMPLETE_MISSING_ACTUAL_ROLLOUTS",
            missing=missing,passed=False,paper_ready=False)

    source=data["source"]
    start=_coords(source,"state_before")
    goal=_coords(source,"goal")
    blockers=[]
    for mode in MODES:
        row=data[mode]
        if int(row.get("seed",-1))!=120:
            raise ValueError(f"{mode}: this DEV gate requires seed 120")
        if (dist(start,_coords(row,"state_before"))>1e-4
                or dist(goal,_coords(row,"goal"))>1e-4):
            raise ValueError(f"{mode}: source and target have different physical start/goal")
        if not isinstance(row.get("physically_verified_matched_success"),bool):
            raise ValueError(f"{mode}: missing actual gate flag")
        if row.get("same_final_goal_config") is not True:
            raise ValueError(f"{mode}: final-goal config not verified")
        if mode!="source":
            blockers.append(_coords(row,"blocker_before"))
    if dist(*blockers)>1e-4:
        raise ValueError("Target direct and clear resets have different blocker positions")

    direct=bool(data["target_direct"]["push_only_success"])
    inserted=bool(data["target_clear"]["blocker_effect_physically_inserted"])
    clear=bool(data["target_clear"]["clear_and_push_success"])
    source_success=bool(source["gate"]["valid"])
    genuinely_measured=all(
        int(data[x].get("steps",data[x].get("actual_rollout_steps",0)))>0
        for x in MODES)
    # A direct-push success means this candidate scene has NOT established
    # that target-only blocker insertion is necessary.
    contrast=bool(source_success and clear and not direct
                  and inserted and genuinely_measured)
    return {
        "type":"M1B2_SAME_GOAL_PHYSX_GATE",
        "status":("PILOT_CONTRAST_OBSERVED_NOT_CAUSAL_PROOF"
                  if contrast else "PILOT_INCONCLUSIVE_OR_FAILED"),
        "source_pick_place_valid":source_success,
        "target_direct_push_success":direct,
        "target_clear_then_push_success":clear,
        "target_only_blocker_displacement_observed":inserted,
        "matched_physics_start_and_goal":True,
        "all_three_real_rollout_traces_present":genuinely_measured,
        "pilot_contrast_pass":contrast,
        "causal_necessity_proven":False,
        "strong_capability_aware_tamp_evaluated":False,
        "learned_method_evaluated":False,
        "paper_ready":False,
        "caution":("Even if direct push fails and clear-then-push succeeds, "
                   "a geometric push-around plan may exist. Prove topology "
                   "and test a strong equal-info TAMP before M2."),
        "per_case":{
            mode:{
                "success":data[mode]["physically_verified_matched_success"],
                "steps":data[mode].get(
                    "steps",data[mode].get("actual_rollout_steps",0)),
                "error":data[mode].get("error"),
            } for mode in MODES
        },
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,
                   default=Path("outputs/m1b2_same_goal_dev"))
    args=p.parse_args(argv)
    out=args.root.resolve()
    result=summarize(out)
    file=out/"physical_gate_summary.json"
    if file.exists() and json.loads(file.read_text())!=result:
        raise FileExistsError(
            f"Existing physical summary differs. Preserve raw run: {file}")
    out.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(result,indent=2)+"\n")
    print("M1B2_PHYSICS_GATE",json.dumps(result),flush=True)


if __name__=="__main__":
    main()
