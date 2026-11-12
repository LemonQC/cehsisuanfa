#!/usr/bin/env bash
# One-Factor v4 on ORIGINAL SAME M1B3 scene: only lower TCP push-contact height.
# Preserve original M0/M1B1/M1B2/M1B3, old v2 450, v3 169 PhysX traces.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

out="outputs/m1_current_target_reference_height_v4"
case="$out/target_reference_height"
mkdir -p "$out"
python -m unittest discover -s tests -p 'test_m1_current_target_reference_height_v4.py' -q
python -m agcofl.m1_current_target_reference_height_v4 --preflight

# Read-only gate: both existing original successful cube motion and
# latest blocked v3 exact contact height failure must be real episodes.
python - <<'PY'
import hashlib,json,math
from pathlib import Path
p0=Path("outputs/m1b3_corridor_dev/open_direct")
p3=Path("outputs/m1_current_target_safe_approach_v3/target_safe_approach")
def load(p):
    a=p/"status.json";b=p/"physical_trace.json"
    if not (a.is_file() and b.is_file()):
        raise SystemExit(f"M1_CONTACT_FRAME_MISSING_OLD_REAL_EVIDENCE: {p}")
    return json.loads(a.read_text()),json.loads(b.read_text()),{
        "status_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),
        "trace_sha256":hashlib.sha256(b.read_bytes()).hexdigest(),
    }
old,trace,old_hash=load(p0)
now,trace3,now_hash=load(p3)
assert old.get("scene")=="AGCOFLM1B3Corridor-v0" and old.get("seed")==120
assert old.get("mode")=="open_direct" and old.get("gate",{}).get("valid") is True
assert old.get("steps")==len(trace)==35
assert now.get("version")=="m1_current_target_safe_approach_v3"
assert now.get("closed_tool_clearance_valid") is True
assert now.get("success") is False and now.get("action_count")==len(trace3)==169
assert "M1_TARGET_NO_VALID_REAR_CONTACT_POSE_BEFORE_PUSH" in now.get("error","")
assert math.dist(old["goal"][:2],now["goal"][:2])<1e-4
start=trace[0]["task_cube"]
moving=[r for r in trace if math.dist(start[:2],r["task_cube"][:2])>.003]
if not moving:
    raise SystemExit("M1_NO_REAL_SUCCESSFUL_CONTACT_REFERENCE")
first=moving[0]
if not (.034<=first["tcp"][2]<=.050 and first["gripper_action"]<0):
    raise SystemExit("M1_SUCCESSFUL_REFERENCE_HEIGHT_NOT_REPRODUCED")
end=trace3[-1]
if (not .034<=end["tcp"][2]<=.050
        or not abs(end["task_cube"][0])<.003
        or not abs(end["tcp"][0]+.055)<.010):
    raise SystemExit("M1_V3_DESIREDCONTACT_MISMATCH")
print("M1_CONTACT_HEIGHT_EXISTING_PHYSICS_GATE",json.dumps({
    "original_succeeded_x_push":True,
    "reference_control_mode":"pd_ee_delta_pose",
    "new_control_mode":"pd_joint_pos",
    "not_same_end_effector_orientation_proof":True,
    "actual_success_contact_tcp_z_m":first["tcp"][2],
    "actual_success_contact_tcp_x_m":first["tcp"][0],
    "v3_stopped_tcp_z_m":end["tcp"][2],
    "v3_stopped_tcp_x_m":end["tcp"][0],
    "v4_derived_tcp_target_z_m":.038,
    "old_open_direct_sha":old_hash,
    "v3_sha":now_hash,
    "new_physics_before_this_gate":0,
}),flush=True)
PY

if [ -s "$case/status.json" ]; then
    echo "M1_REFERENCE_HEIGHT_IMMUTABLE_RESULT_REUSED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_REFERENCE_HEIGHT_PARTIAL_RESULT_PRESERVE_NO_OVERWRITE" >&2
    exit 5
else
    echo "M1_REFERENCE_HEIGHT_ONE_BOUNDED_PHYSICS_TRIAL"
    timeout --signal=TERM --kill-after=10s 500s \
        python -u -m agcofl.m1_current_target_reference_height_v4 \
        --out "$out" | tee "$out/reference_height_v4_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_REFERENCE_HEIGHT_MISSING_STATUS")
r=json.loads(p.read_text())
print("M1_REFERENCE_HEIGHT_GATE",json.dumps({
    "success":r.get("success"),
    "actions":r.get("action_count"),
    "clearance_valid":r.get("closed_tool_clearance_valid"),
    "goal_error_m":r.get("cube_goal_error_m"),
    "cube_final_xyz_m":r.get("cube_final"),
    "actual_contact_tcp_xyz_m":r.get("contact_tcp_real_after_descent_xyz_m"),
    "guard_violation":r.get("guard_violation"),
    "error":r.get("error"),
    "phases":r.get("phases"),
    "source_same_goal_valid":False,
    "new_method_or_tamp_evaluated":False
}),flush=True)
PY
