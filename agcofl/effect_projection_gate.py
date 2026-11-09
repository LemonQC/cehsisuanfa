"""First science-facing gate: source 3D object effect vs Push-only capability.

Geometry + state evidence ONLY. This must NEVER label a path physically
reachable until a robot has actually executed it in the simulator.
For supervised paper experiments, inputs must be calibrated SE(3) tracks.
"""
from __future__ import annotations
import argparse
import heapq
import json
import math
from pathlib import Path
import numpy as np


def _point_collision(p, boxes, radius):
    x,y=float(p[0]),float(p[1])
    return any(b[0]-radius <=x<=b[2]+radius and
               b[1]-radius <=y<=b[3]+radius for b in boxes)


def line_free(p,q,boxes,radius,spacing=.008):
    a=np.asarray(p,float)[:2];b=np.asarray(q,float)[:2]
    distance=np.linalg.norm(b-a)
    for t in np.linspace(0,1,max(2,int(np.ceil(distance/spacing))+1)):
        if _point_collision(a+(b-a)*t,boxes,radius):return False
    return True


def path_length(path):
    a=np.asarray(path,float)
    return float(np.linalg.norm(np.diff(a,axis=0),axis=-1).sum()) if len(a)>1 else 0.


def astar_xy(start,goal,boxes,bounds=(-.35,.35,-.35,.35),
             resolution=.01,radius=.028):
    """Deterministic grid candidate. Not a robot dynamics or contact solver."""
    start=np.asarray(start,float)[:2];goal=np.asarray(goal,float)[:2]
    xmin,xmax,ymin,ymax=map(float,bounds)
    if not (xmin<=start[0]<=xmax and ymin<=start[1]<=ymax and
            xmin<=goal[0]<=xmax and ymin<=goal[1]<=ymax):
        return None
    if (_point_collision(start,boxes,radius) or
        _point_collision(goal,boxes,radius)):
        return None
    if line_free(start,goal,boxes,radius):
        return [start.tolist(),goal.tolist()]
    nx=int(np.floor((xmax-xmin)/resolution))+1
    ny=int(np.floor((ymax-ymin)/resolution))+1
    if nx*ny>150000:
        raise ValueError("Grid exceeds 150000 nodes; increase resolution")
    def key(x):
        return (int(round((x[0]-xmin)/resolution)),
                int(round((x[1]-ymin)/resolution)))
    def xy(k):return np.array([xmin+k[0]*resolution,ymin+k[1]*resolution])
    a=key(start);b=key(goal)
    valid=lambda k: 0<=k[0]<nx and 0<=k[1]<ny and not _point_collision(xy(k),boxes,radius)
    if not(valid(a) and valid(b)):return None
    moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    open_set=[(float(np.linalg.norm(start-goal)),0.,a)]
    parent={};best={a:0.};visited=set()
    while open_set:
        _,cost,u=heapq.heappop(open_set)
        if u in visited:continue
        visited.add(u)
        if u==b:
            nodes=[u]
            while nodes[-1]!=a:nodes.append(parent[nodes[-1]])
            nodes.reverse()
            raw=[start]+[xy(k) for k in nodes[1:-1]]+[goal]
            # Remove geometrically redundant intermediate grid cells, without
            # deleting a task-specified waypoint (those are segmented below).
            smooth=[raw[0]]
            i=0
            while i<len(raw)-1:
                nxt=i+1
                for j in range(len(raw)-1,i,-1):
                    if line_free(raw[i],raw[j],boxes,radius):
                        nxt=j
                        break
                smooth.append(raw[nxt]);i=nxt
            return [v.tolist() for v in smooth]
        for dx,dy in moves:
            v=(u[0]+dx,u[1]+dy)
            if not valid(v) or v in visited:continue
            if not line_free(xy(u),xy(v),boxes,radius):continue
            step=resolution*math.hypot(dx,dy)
            score=cost+step
            if score+1e-10<best.get(v,float("inf")):
                best[v]=score;parent[v]=u
                heapq.heappush(open_set,
                    (score+float(np.linalg.norm(xy(v)-goal)),score,v))
    return None


def ordered_path(start,goal,via,boxes,bounds,resolution,radius):
    route=[np.asarray(start,float).tolist()]
    stops=[np.asarray(v,float)[:2] for v in via]+[np.asarray(goal,float)[:2]]
    for target in stops:
        segment=astar_xy(route[-1],target,boxes,bounds,resolution,radius)
        if segment is None:return None
        route.extend(segment[1:])
    return route


def _close_to_path(path,waypoint,radius=.05):
    if path is None:return False
    for p,q in zip(path[:-1],path[1:]):
        a=np.asarray(p,float);b=np.asarray(q,float);v=b-a
        t=float(np.clip(np.dot(np.asarray(waypoint)-a,v)/
                        max(np.dot(v,v),1e-10),0.,1.))
        if np.linalg.norm(a+t*v-np.asarray(waypoint))<=radius:
            return True
    return False


def analyze_effect(data):
    """Reject unjustified 3D claims from 2D MP4 pseudo-segmentation."""
    if "source_object_xyz" not in data:
        raise ValueError("Need calibrated source_object_xyz [N,3] (not video pixel XY)")
    xyz=np.asarray(data["source_object_xyz"],float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
        raise ValueError("source_object_xyz must be finite [N>=3,3]")
    required=["table_center_z","goal_xy","bounds_xy","obstacles_xyxy",
              "source_provenance","source_mode"]
    missing=[k for k in required if k not in data]
    if missing:raise ValueError("Missing fields: "+repr(missing))
    table_z=float(data["table_center_z"])
    ztol=float(data.get("support_z_tolerance",.015))
    radius=float(data.get("object_xy_radius",.028))
    boxes=data["obstacles_xyxy"]
    if any(len(b)!=4 or b[0]>=b[2] or b[1]>=b[3] for b in boxes):
        raise ValueError("obstacles_xyxy must contain minx,miny,maxx,maxy rectangles")
    start=xyz[0,:2].tolist();goal=list(map(float,data["goal_xy"]))
    via=data.get("required_via_xy",[])
    if any(len(v)!=2 for v in via):raise ValueError("required_via_xy entries must be XY")
    bounds=data["bounds_xy"]
    if len(bounds)!=4:raise ValueError("bounds_xy must be minx,maxx,miny,maxy")
    res=float(data.get("grid_resolution",.012))
    support_violations=int((abs(xyz[:,2]-table_z)>ztol).sum())
    flat_collision_count=sum(_point_collision(x,boxes,radius) for x in xyz)
    source_mode=str(data["source_mode"]).lower()
    allowed_modes=("push","pick","unknown")
    if source_mode not in allowed_modes:
        raise ValueError(f"source_mode must be in {allowed_modes}")
    # In a source Pick video, the object's XY path may be above a physical
    # obstacle. Flattening it for Push without re-planning is UNSAFE.
    mode_matches=(source_mode=="push")
    mode_known=(source_mode in ("push","pick"))
    directly_compatible=(support_violations==0 and
                          flat_collision_count==0 and mode_matches and
                          all(_close_to_path(xyz[:,:2],p) for p in via))
    goal_only=ordered_path(start,goal,[],boxes,bounds,res,radius)
    oracle=ordered_path(start,goal,via,boxes,bounds,res,radius)
    projected=ordered_path(start,goal,via,boxes,bounds,res,radius)
    # Deliberately the same geometric construction as waypoint oracle:
    # without a learned capability model or physics objective, there is no
    # algorithmic novelty to claim, even if this path exists.
    candidate_status=("DIRECTLY_GEOMETRICALLY_COMPATIBLE_PENDING_CONTACT_VALIDATION"
                      if directly_compatible else
                      "GEOMETRIC_PROJECTABLE_PENDING_PHYSICS"
                      if projected is not None else
                      "UNRESOLVED_WITHIN_GRID_BUDGET")
    if not mode_known and projected is not None:
        candidate_status="SOURCE_MODE_UNKNOWN_PHYSICS_VALIDATION_REQUIRED"
    def desc(path):
        return dict(exists=path is not None,
            length_m=None if path is None else round(path_length(path),5),
            meets_waypoints=all(_close_to_path(path,p) for p in via),
            collision_free=path is not None and
              all(line_free(a,b,boxes,radius) for a,b in zip(path[:-1],path[1:])),
            path_xy=path)
    result=dict(
        type="GEOMETRIC_EFFECT_CANDIDATE_NOT_PHYSICS_PROOF",
        source_provenance=data["source_provenance"],source_mode=source_mode,
        source_points=len(xyz),max_source_height_m=float(np.max(xyz[:,2]-table_z)),
        support_violation_frames=support_violations,
        push_projected_xy_collision_frames=int(flat_collision_count),
        requested_waypoints=via,goal_xy=goal,
        source_directly_compatible_by_geometry=bool(directly_compatible),
        candidate_status=candidate_status,
        baselines=dict(source_raw=dict(
             support_valid=support_violations==0,
             push_mode_valid=mode_matches,
             projected_xy_collision_free=flat_collision_count==0,
             specified_waypoints=all(_close_to_path(xyz[:,:2],p) for p in via)),
            goal_only=desc(goal_only),
            goal_waypoint_oracle=desc(oracle)),
        proposed_geometric_candidate=desc(projected),
        hard_goal_constraint=projected is not None,
        identical_to_waypoint_oracle=projected==oracle,
        disclaimer=("Geometry checks do not verify robot joint reachability, "
                    "contact force, slip, collision during approach, or task success. "
                    "A proposed method identical to Goal+Waypoint Oracle has "
                    "NO algorithmic improvement; physics rollout and learned "
                    "source-effect extraction are required for T-RO claims.")
    )
    return result


def from_saved_robot_physics(ep_dir):
    """Uses the user's ACTUAL 31-step PhysX PushCube trace, NOT an invented Pick source."""
    ep_dir=Path(ep_dir)
    track=json.loads((ep_dir/"physical_trace.json").read_text())
    outcome=json.loads((ep_dir/"episode_result.json").read_text())
    points=[p["cube"] for p in track]+[outcome["final_cube"]]
    xy=np.asarray(points)[:,:2]
    pad=.12
    bounds=[float(min(xy[:,0].min(),outcome["final_goal"][0])-pad),
            float(max(xy[:,0].max(),outcome["final_goal"][0])+pad),
            float(min(xy[:,1].min(),outcome["final_goal"][1])-pad),
            float(max(xy[:,1].max(),outcome["final_goal"][1])+pad)]
    return dict(source_object_xyz=points,source_mode="unknown",
       source_provenance={"type":"REAL_SIMULATOR_OBJECT_STATE_NOT_SOURCE_PICK",
         "path":str((ep_dir/"physical_trace.json").resolve()),
         "warning":"Object XYZ is real PhysX. Contact mode was NOT measured, never infer push contact from position alone."},
       table_center_z=.02,goal_xy=outcome["final_goal"][:2],
       bounds_xy=bounds,obstacles_xyxy=[],
       required_via_xy=[],object_xy_radius=.028)


def main(argv=None):
    p=argparse.ArgumentParser(description="3D object-effect geometry / Push-only capability gate")
    x=p.add_mutually_exclusive_group(required=True)
    x.add_argument("--input",help="JSON calibrated 3D source Pick effect")
    x.add_argument("--from-saved-robot-trace",help="Existing PhysX Push trace: adapter sanity, NOT cross-behavior")
    p.add_argument("--out",default="outputs/effect_projection_gate/gate_report.json")
    args=p.parse_args(argv)
    data=(json.loads(Path(args.input).read_text()) if args.input
          else from_saved_robot_physics(args.from_saved_robot_trace))
    report=analyze_effect(data)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print("EFFECT_GEOMETRY_GATE",json.dumps({
       "source":report["source_provenance"].get("type",str(report["source_provenance"])),
       "status":report["candidate_status"],
       "source_support_violations":report["support_violation_frames"],
       "source_projected_collisions":report["push_projected_xy_collision_frames"],
       "projected_path_exists":report["proposed_geometric_candidate"]["exists"],
       "identical_to_waypoint_oracle":report["identical_to_waypoint_oracle"],
       "warning":report["type"]
    }),flush=True)
    return report


if __name__=="__main__":main()
