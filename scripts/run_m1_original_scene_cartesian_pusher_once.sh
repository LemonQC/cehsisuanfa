#!/usr/bin/env bash
# ONE current M1 same-scene target actuator swap. No changed geometry,
# old data overwrite or 450-step blind Screw retry.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1_original_scene_cartesian_pusher_v5"
case="$out/target_cartesian_pusher"
mkdir -p "$out"

python -m unittest discover -s tests \
  -p 'test_m1_original_scene_cartesian_pusher_v5.py' -q
python -m agcofl.m1_original_scene_cartesian_pusher_v5 --preflight

# Earlier physical evidence is read only: 35-step original
# pd_ee_delta_pose successfully pushes x=.21, and v4 pd_joint_pos
# Screw stage3 jams at X=.131 despite 210 actual actions.
python - <<'PY'
import hashlib,json
from pathlib import Path
old=Path("outputs/m1b3_corridor_dev/open_direct")
stuck=Path("outputs/m1_current_target_reference_height_v4/target_reference_height")
def read(p):
    a=p/"status.json";b=p/"physical_trace.json"
    if not (a.is_file() and b.is_file()):
        raise SystemExit(f"M1_V5_MISSING_FROZEN_REAL_PHYSICS_EVIDENCE {p}")
    st=json.loads(a.read_text());rs=json.loads(b.read_text())
    return st,rs,{
        "status_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),
        "trace_sha256":hashlib.sha256(b.read_bytes()).hexdigest()
    }
a,ra,ah=read(old)
b,rb,bh=read(stuck)
assert a.get("mode")=="open_direct" and a.get("scene")=="AGCOFLM1B3Corridor-v0"
assert a.get("gate",{}).get("valid") is True and a.get("steps")==len(ra)==35
assert all(x.get("gripper_action")==-1. for x in ra)
assert b.get("kind")=="target_reference_height" and b.get("success") is False
assert b.get("action_count")==len(rb)==450 and b.get("seed")==120
third=[r for r in rb if r.get("execution_phase")=="push_3_x_0.260"]
assert len(third)==210
tcp_dx=max(r["tcp"][0] for r in third)-min(r["tcp"][0] for r in third)
cube_dx=max(r["task_cube"][0] for r in third)-min(r["task_cube"][0] for r in third)
if not (tcp_dx<.001 and cube_dx<.001 and
        b.get("closed_tool_clearance_valid") is True):
    raise SystemExit("M1_V5_NOT_PROVEN_REAL_TCP_STALL_DO_NOT_RUN")
assert abs(a["goal"][0] - b["goal"][0])<.0001
assert abs(a["state_before"][0] - b["cube_initial"][0])<.0001
print("M1_V5_REAL_EXISTING_PHYSICS_GATE",json.dumps({
 "original_cartesian_success_35_actions":True,
 "old_jointspace_210_action_tcp_range_m":tcp_dx,
 "old_jointspace_210_action_cube_range_m":cube_dx,
 "same_original_goal_and_cube_start":True,
 "original_control_mode":"pd_ee_delta_pose",
 "old_failed_control_mode":"pd_joint_pos",
 "raw_open_sha256":ah,
 "raw_v4_sha256":bh,
 "new_physics_yet":False,
}),flush=True)
PY

if [ -s "$case/status.json" ]; then
    echo "M1_V5_CARTESIAN_EXISTING_PHYSICS_PRESERVED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_V5_PARTIAL_RAW_EVIDENCE_NO_OVERWRITE" >&2
    exit 5
else
    echo "M1_V5_CARTESIAN_ONE_BOUNDED_PHYSX_ATTEMPT_SAME_SCENE"
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1_original_scene_cartesian_pusher_v5 \
        --out "$out" | tee "$out/run_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_V5_REAL_STATUS_MISSING")
r=json.loads(p.read_text())
print("M1_V5_CARTESIAN_PHYSICS_GATE",json.dumps({
 "success":r.get("success"),
 "steps":r.get("steps"),
 "clearance_valid":r.get("clearance_valid"),
 "cube_final_xyz_m":r.get("cube_final"),
 "goal_error_m":r.get("goal_error_m"),
 "phases":r.get("phases"),
 "error":r.get("error"),
 "source_same_goal_valid":False,
 "equal_information_tamp_or_learned_method_evaluated":False,
}),flush=True)
PY
