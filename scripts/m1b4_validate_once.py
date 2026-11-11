#!/usr/bin/env python3
"""Check at most two genuine M1B4 rescue episodes. Never claim TAMP/novelty."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CASES = ("source_lower_fallback", "target_align_after_clear")


def _vector(obj, name):
    data=obj.get(name)
    if (not isinstance(data,list) or len(data)!=3
            or any(type(q) not in (int,float) or not math.isfinite(q)
                   for q in data)):
        raise ValueError(f"{name}: invalid real 3D state")
    return data


def validate(root):
    entries={}
    for mode in CASES:
        file=root/mode/"status.json"
        trace=root/mode/"physical_trace.json"
        if not file.is_file() or not trace.is_file():
            raise FileNotFoundError(f"Original M1B4 real physical evidence missing: {mode}")
        result=json.loads(file.read_text())
        steps=json.loads(trace.read_text())
        if (result.get("mode")!=mode
            or result.get("version")!="m1b4_same_geometry_v1_execution_rescue"
            or result.get("original_scene")!="AGCOFLM1B3Corridor-v0"
            or result.get("geometry_unchanged") is not True
            or result.get("seed")!=120
            or type(result.get("steps")) is not int
            or result["steps"]<=0
            or len(steps)!=result["steps"]):
            raise ValueError(f"{mode}: not an actual validated matching-version trajectory")
        for key in ("state_before","goal","final"):
            _vector(result,key)
        if (not isinstance(result.get("gate"),dict)
                or type(result["gate"].get("valid")) is not bool):
            raise ValueError(f"{mode}: no measured success label")
        entries[mode]=result
    a,b=(entries[q] for q in CASES)
    if any(math.dist(a[k],b[k])>.0001 for k in ("state_before","goal")):
        raise ValueError("source and target final goal/initial cube state differ")
    return {
        "type":"M1B4_ONE_TIME_FEASIBILITY_GATE",
        "status":("PHYSICS_PAIR_FEASIBLE_NOT_SCIENTIFIC_NOVELTY"
                  if a["gate"]["valid"] and b["gate"]["valid"]
                  else "PAIRED_PHYSICS_STILL_FAILED_STOP_THIS_SCENE"),
        "source_success":a["gate"]["valid"],
        "target_success":b["gate"]["valid"],
        "same_initial_cube_and_goal":True,
        "unchanged_old_geometry":True,
        "new_experiments":2,
        "source_steps":a["steps"],"target_steps":b["steps"],
        "source_error":a.get("error"),"target_error":b.get("error"),
        "strong_tamp_evaluated":False,
        "learned_method_evaluated":False,
        "source_video_information_measured":False,
        "causal_necessity_proven":False,
        "paper_ready":False,
        "stop_if_failed":True,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,
                   default=Path("outputs/m1b4_one_time_feasibility"))
    args=p.parse_args(argv)
    root=args.out.resolve()
    data=validate(root)
    file=root/"gate.json"
    content=json.dumps(data,indent=2)+"\n"
    if file.exists() and file.read_text()!=content:
        raise FileExistsError("Existing one-time result differs; preserve all evidence")
    file.write_text(content)
    print("M1B4_FEASIBILITY_GATE",json.dumps(data),flush=True)


if __name__=="__main__":
    main()
