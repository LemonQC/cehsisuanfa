"""Paired post-hoc diagnostic of RoboReel fixed vs video-motion controller."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def compare(root):
    root=Path(root)
    base=root/'fixed'/'report.json'
    other=root/'video_motion'/'report.json'
    if not base.is_file() or not other.is_file():
        raise FileNotFoundError("Both fixed/report.json and video_motion/report.json are required")
    a=json.loads(base.read_text())
    b=json.loads(other.read_text())
    left={r["episode_id"]:r for r in a["results"]}
    right={r["episode_id"]:r for r in b["results"]}
    ids=sorted(set(left)&set(right))
    if not ids:raise ValueError("No matched episodes")
    rows=[]
    for eid in ids:
        fixed,video=left[eid],right[eid]
        if fixed["human_demo_id"]!=video["human_demo_id"]:
            raise ValueError("Human videos not paired on the same episode")
        fa=np.load(root/'fixed'/f'episode_{eid:04d}'/'actions.npy')
        va=np.load(root/'video_motion'/f'episode_{eid:04d}'/'actions.npy')
        n=min(len(fa),len(va))
        diff=float(np.mean(np.linalg.norm(fa[:n]-va[:n],axis=1)))
        rows.append(dict(episode_id=eid,human_demo_id=fixed["human_demo_id"],
            fixed_success=int(fixed["success"]),video_success=int(video["success"]),
            fixed_steps=fixed["steps"],video_steps=video["steps"],
            fixed_goal_error=fixed["final_goal_error_m"],
            video_goal_error=video["final_goal_error_m"],
            video_gain=video["video_gain"],
            mean_paired_action_difference=diff,
            detected_video_coverage=video["video_meta"]["coverage"]))
    with (root/'paired_diagnostic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    result=dict(
        type='PRIVILEGED_CONTROLLER_VIDEO_SCALAR_DIAGNOSTIC_ONLY',
        episodes=len(rows),fixed_successes=sum(r["fixed_success"] for r in rows),
        video_successes=sum(r["video_success"] for r in rows),
        mean_paired_action_difference=float(np.mean([r["mean_paired_action_difference"] for r in rows])),
        rows=rows,
        conclusion_limit='Different actions/success here are not Full/Anchor/End supervision evidence, because the controller uses privileged simulator object/goal state.'
    )
    (root/'paired_diagnostic.json').write_text(json.dumps(result,indent=2))
    print('ROB OREEL_PAIRED_DIAGNOSTIC'.replace(' ',''),json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='outputs/roboreel_real')
    a=p.parse_args()
    compare(a.out)
