#!/usr/bin/env bash
# ONE bounded correction of a demonstrated acceptance bug, ORIGINAL M1B3,
# not repeat v5 and not another new task/model. Preserve all previous raw.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1_original_scene_cartesian_pusher_v6"
case="$out/target_cartesian_contact_fixed"
mkdir -p "$out"

python -m unittest discover -s tests \
  -p 'test_m1_original_scene_cartesian_pusher_v6.py' -q
python -m agcofl.m1_original_scene_cartesian_pusher_v6 --preflight

# Read only the exact actual failed v5 case. Do NOT relabel mocked tests
# or previous failed physics. No simulation if the evidence has changed.
python - <<'PY'
import hashlib,json,math
from pathlib import Path
root=Path("outputs/m1_original_scene_cartesian_pusher_v5/target_cartesian_pusher")
a=root/"status.json";b=root/"physical_trace.json"
if not (a.is_file() and b.is_file()):
    raise SystemExit("M1_V6_MISSING_REAL_V5_54_STEP_EVIDENCE")
s=json.loads(a.read_text());trace=json.loads(b.read_text())
assert s.get("version")=="m1_original_scene_cartesian_pusher_v5"
assert s.get("success") is False and s.get("steps")==len(trace)==54
assert s.get("clearance_valid") is True
assert s.get("error")=="RuntimeError: ORIGINAL_PUSH_CONTACT_POSE_INVALID"
assert s.get("scene")=="AGCOFLM1B3Corridor-v0" and s.get("seed")==120
assert abs(s["goal"][0]-.210)<1e-5
phases=s.get("phases",[])
assert [r["phase"] for r in phases]==[
 "clear_blocker","retreat_vertical","hover_rear","descend_rear"
]
assert [r["actions"] for r in phases]==[32,7,9,6]
last=trace[-1];tcp=last["tcp"]
commanded=[-.060,0.,.038]
res=math.dist(tcp,commanded)
assert .016<res<.022, f"v5 premature stop not demonstrated: {res}"
assert tcp[2]>.052 and tcp[0]<-.028
assert abs(last["task_cube"][0])<.003 and last["gripper_action"]<0
print("M1_V6_REAL_PREMATURE_DESCENT_EVIDENCE",json.dumps({
 "v5_actual_steps":54,
 "old_descent_last_tcp_xyz_m":tcp,
 "old_descent_target_tcp_xyz_m":commanded,
 "old_descent_actual_3d_error_m":res,
 "v5_wrong_approach_tolerance_m":.022,
 "v6_single_fixed_tolerance_m":.008,
 "original_scene_and_clearance_unchanged":True,
 "v5_source_sha256":{
  "status":hashlib.sha256(a.read_bytes()).hexdigest(),
  "trace":hashlib.sha256(b.read_bytes()).hexdigest(),
 },
 "new_physics_yet":False,
}),flush=True)
PY

if [ -s "$case/status.json" ]; then
    echo "M1_V6_EXISTING_PHYSICAL_EVIDENCE_PRESERVED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_V6_PARTIAL_RAW_PRESERVED_STOP" >&2
    exit 5
else
    echo "M1_V6_SINGLE_BOUNDED_ORIGINAL_SCENE_PHYSX"
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1_original_scene_cartesian_pusher_v6 \
        --out "$out" | tee "$out/v6_run_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    raise SystemExit("M1_V6_MISSING_REAL_STATUS")
r=json.loads(p.read_text())
print("M1_V6_PHYSICAL_GATE",json.dumps({
 "success":r.get("success"),
 "steps":r.get("steps"),
 "clearance_valid":r.get("clearance_valid"),
 "cube_final_xyz_m":r.get("cube_final"),
 "goal_error_m":r.get("goal_error_m"),
 "phases":r.get("phases"),
 "push_only":r.get("push_only"),
 "error":r.get("error"),
 "same_goal_source_pick":False,
 "learned_method_or_tamp_evaluated":False,
}),flush=True)
PY
