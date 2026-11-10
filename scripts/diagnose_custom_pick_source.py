#!/usr/bin/env python3
"""Offline, read-only diagnosis of existing CUSTOM Pick source PhysX trace.

Never launches ManiSkill or a GPU; never modifies the existing run. The source
state machine can enter "failed" ONLY after exceeding the grasp-phase timeout;
ever_grasped is measured separately after env.step in ALL phases and does
not establish sustained grasp while in the 'grasp' phase.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

DEFAULT = Path("outputs/v140_custom_pick_push_dev/source_pick")

def analyze(summary, trace):
    counts=Counter(str(row["phase"]) for row in trace)
    phases=[]
    for phase in dict.fromkeys(str(r["phase"]) for r in trace):
        rows=[r for r in trace if r["phase"]==phase]
        starts=np.asarray([r["cube"] for r in rows],dtype=float)
        tcp=np.asarray([r["tcp"] for r in rows],dtype=float)
        perceived=[bool(r["is_grasped"]) for r in rows]
        phases.append({
            "phase":phase,"steps":len(rows),
            "grasp_detected_pre_action":int(sum(perceived)),
            "max_consecutive_grasp_pre_action":_longest_true(perceived),
            "first_tcp_xyz":tcp[0].tolist(),
            "last_tcp_xyz":tcp[-1].tolist(),
            "tcp_displacement_m":float(np.linalg.norm(tcp[-1]-tcp[0])),
            "first_cube_xyz":starts[0].tolist(),
            "last_cube_xyz":starts[-1].tolist(),
            "max_cube_z_m":float(starts[:,2].max()),
            "max_cube_xy_displacement_from_phase_start_m":
                float(np.linalg.norm(starts[:,:2]-starts[0,:2],axis=1).max()),
            "gripper_close_command_steps":sum(float(r["gripper_command"])>0 for r in rows),
        })
    return {
        "type":"CUSTOM_PICK_SOURCE_TRACE_OFFLINE_DIAGNOSTIC",
        "seed":summary.get("seed"),"source_steps":len(trace),
        "final_phase":summary.get("end_phase"),"reported_ever_grasped":summary.get("ever_grasped"),
        "reported_lift_m":summary.get("peak_lift_m"),
        "reported_source_valid":summary.get("valid_pick_source"),
        "phase_counts":dict(counts),"phase_diagnostics":phases,
        "no_physics_executed":True,
        "interpretation":(
            "End phase 'failed' is reached only from 'grasp' timeout in current "
            "source state machine. 'ever_grasped' can be true in a different "
            "phase/post-action sample; inspect phase_diagnostics for sustained "
            "grasp and relative TCP/cube motion. The trace samples before each "
            "action; post-action transient is not necessarily represented."
        )
    }

def _longest_true(values):
    best=run=0
    for x in values:
        run=run+1 if x else 0
        best=max(best,run)
    return best

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=DEFAULT)
    args=parser.parse_args(argv)
    summary=json.loads((args.directory/"source_summary.json").read_text())
    trace=json.loads((args.directory/"source_trace.json").read_text())
    report=analyze(summary,trace)
    print("CUSTOM_PICK_SOURCE_DIAG",json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
