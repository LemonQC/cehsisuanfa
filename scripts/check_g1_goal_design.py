#!/usr/bin/env python3
"""Offline ONLY: sample deterministic dual goal geometry, never simulate physics.

G0 human video audit is still BLOCKED until real video review. This tool
cannot authorize real rollouts or prove Panda reachability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from agcofl.video_goal_push_spec import CounterfactualPushSpec, assert_no_goal_leak


def build_design(seeds:tuple[int,...]=(310000,310001)) -> dict:
    spec=CounterfactualPushSpec()
    rows=[]
    for seed in seeds:
        rng=random.Random(seed)
        init=(rng.uniform(-0.10,0.10),rng.uniform(-0.10,0.10),spec.cube_half_size_m)
        paired=spec.paired_design(init,seed)
        # Policy has exactly the same initial input in both evaluator branches:
        obs={"rgb": "same-camera-observation", "robot_joint_state": "same-reset"}
        assert_no_goal_leak(obs)
        paired["policy_observation_structure"] = list(obs)
        paired["policy_input_equal_across_branches_by_construction"] = True
        rows.append(paired)
    return {"protocol":"TRO1_G1_GEOMETRY_DESIGN_DRAFT_V0",
            "g0_real_human_video_pair_verified":False,
            "new_physx_actions":0,"new_training_runs":0,
            "model_inference_executed":False,"physical_reachability_verified":False,
            "goal_evaluator_only":True,"physical_task_registration":False,
            "design_goal_pairs":rows,"decision":"G1_GEOMETRY_ONLY_G0_BLOCKED"}


def run(project:Path,out:Path) -> Path:
    d=build_design()
    payload=json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
    digest=hashlib.sha256(payload.encode()).hexdigest()[:16]
    target=out/f"design_{digest}.json"
    if not target.resolve().is_relative_to(project.resolve()):
        raise ValueError("OUTPUT_OUTSIDE_PROJECT")
    out.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_text()!=payload:
            raise FileExistsError("DO_NOT_OVERWRITE_CHANGED_GEOMETRY_DESIGN")
    else:
        target.write_text(payload)
    return target


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,default=Path("outputs/video_goal_push_v0/geometry_gate"))
    args=p.parse_args(argv)
    project=args.project.resolve()
    out=args.out if args.out.is_absolute() else project/args.out
    dest=run(project,out)
    print("TRO1_G1_GEOMETRY_DESIGN",json.dumps({"status":"GEOMETRY_DESIGN_READY_PHYSICS_NOT_RUN",
        "g0":"BLOCKED_NO_VERIFIED_HUMAN_PAIR","report":str(dest),
        "new_physx_actions":0,"new_training_runs":0},ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
