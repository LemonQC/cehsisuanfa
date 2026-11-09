"""Create watchable MP4 from REAL RoboReel physics trace, without rendering GPU.

This is an overhead 2D state replay (cube, TCP, goal), NOT an RGB robot video.
It uses the saved per-step physical values and does NOT simulate new actions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def render_trace(episode_dir, output=None, fps=12):
    import cv2

    episode_dir = Path(episode_dir)
    trace_path = episode_dir / "physical_trace.json"
    result_path = episode_dir / "episode_result.json"
    if not trace_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("Need physical_trace.json and episode_result.json in " + str(episode_dir))
    trace = json.loads(trace_path.read_text())
    result = json.loads(result_path.read_text())
    if not trace:
        raise ValueError("Empty physics trace")
    xy = np.asarray([p["cube"][:2] for p in trace] +
                    [result["final_cube"][:2]], np.float32)
    tcp = np.asarray([p["tcp"][:2] for p in trace], np.float32)
    goal = np.asarray(result["final_goal"][:2], np.float32)
    if not (np.isfinite(xy).all() and np.isfinite(tcp).all() and np.isfinite(goal).all()):
        raise ValueError("Invalid non-finite simulator trace")

    W, H = 720, 540
    margin = 80
    cloud = np.concatenate([xy, tcp, goal[None]], axis=0)
    low = np.min(cloud, axis=0) - .08
    high = np.max(cloud, axis=0) + .08
    ranges = np.maximum(high-low, .15)
    scale = float(min((W-2*margin)/ranges[0], (H-2*margin)/ranges[1]))
    center = (low+high)/2.

    def pixel(p):
        v = np.asarray(p, np.float32)
        return (int(round(W/2+(v[0]-center[0])*scale)),
                int(round(H/2-(v[1]-center[1])*scale)))

    output = Path(output) if output is not None else episode_dir/"physical_xy_replay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 encoder could not open output: "+str(output))

    last_frame = None
    try:
        for i in range(len(xy)):
            frame = np.full((H,W,3), (245,247,250), dtype=np.uint8)
            # simple world axes; this is NOT a rendered ManiSkill camera
            cv2.line(frame, pixel([0,low[1]]), pixel([0,high[1]]),(218,215,210),1)
            cv2.line(frame, pixel([low[0],0]), pixel([high[0],0]),(218,215,210),1)
            target = pixel(goal)
            cv2.circle(frame,target,round(.05*scale),(150,215,160),2)
            cv2.circle(frame,target,6,(40,155,45),-1)
            if i>0:
                pts=np.asarray([pixel(v) for v in xy[:i+1]],np.int32)
                cv2.polylines(frame,[pts],False,(180,135,45),2)
            cube_pos=pixel(xy[i])
            side=max(5,int(round(.02*scale)))
            cv2.rectangle(frame,(cube_pos[0]-side,cube_pos[1]-side),
                          (cube_pos[0]+side,cube_pos[1]+side),(215,95,45),-1)
            if i<len(tcp):
                tip=pixel(tcp[i])
                cv2.circle(frame,tip,6,(155,35,145),-1)
                phase=trace[i]["phase"]
            else:
                phase="final"
            cv2.putText(frame,"RoboReel PushCube | actual simulator XY trace",
                        (22,30),cv2.FONT_HERSHEY_SIMPLEX,.66,(42,42,42),2)
            cv2.putText(frame,"NOT RGB robot video | saved physics measurements",
                        (22,54),cv2.FONT_HERSHEY_SIMPLEX,.53,(80,80,80),1)
            cv2.putText(frame,f"step {i}/{len(xy)-1}   phase: {phase}",
                        (22,H-56),cv2.FONT_HERSHEY_SIMPLEX,.55,(45,45,45),1)
            error=float(np.linalg.norm(xy[i]-goal))
            cv2.putText(frame,f"goal error: {error*100:.2f} cm   success: {result['success']}",
                        (22,H-30),cv2.FONT_HERSHEY_SIMPLEX,.55,(45,45,45),1)
            cv2.putText(frame,"blue=cube  purple=TCP  green=goal",
                        (370,H-30),cv2.FONT_HERSHEY_SIMPLEX,.45,(70,70,70),1)
            writer.write(frame)
            last_frame=frame
    finally:
        writer.release()
    if last_frame is not None:
        cv2.imwrite(str(output.with_suffix(".png")),last_frame)
    summary={"source_trace":str(trace_path), "source_result":str(result_path),
             "video":str(output),"frames":len(xy),"fps":fps,
             "label":"PHYSICS_XY_TRACE_REPLAY_NOT_RGB_SIMULATION",
             "success_from_saved_result":bool(result["success"]),
             "end_error_m_from_saved_result":float(result["final_goal_error_m"])}
    (output.with_suffix(".json")).write_text(json.dumps(summary,indent=2))
    print("PHYSICS_REPLAY_WRITTEN",json.dumps(summary),flush=True)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episode-dir",
       default="outputs/roboreel_cn/fixed/episode_0000")
    p.add_argument("--out",default=None)
    p.add_argument("--fps",type=int,default=12)
    a=p.parse_args()
    render_trace(a.episode_dir,a.out,a.fps)


if __name__=="__main__":
    main()
