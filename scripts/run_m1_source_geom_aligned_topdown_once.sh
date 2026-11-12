#!/usr/bin/env bash
# ONE guarded, original-scene source Pick contact/transfer PhysX candidate.
# Panda actual collision envelope and three frozen old source traces must
# verify before a SINGLE trial. Target v6 66-step success is FROZEN.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

out="outputs/m1_source_geom_aligned_topdown_v1"
case="$out/source_aligned_topdown"
mkdir -p "$out"

python -m unittest discover -s tests \
  -p 'test_m1_source_geom_aligned_topdown_v1.py' -q

python -m agcofl.m1_source_geom_aligned_topdown_v1 --preflight

# STRICT zero-PhysX 3-way real evidence gate and installed Panda URDF.
# Never use original B2 x=.15 as a matched x=.21 source.
python - <<'PY'
import json
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from mani_skill import PACKAGE_ASSET_DIR
from agcofl.m1_source_geom_aligned_topdown_v1 import geometry_contract
from scripts.m1_source_three_way_existing_audit import analyse
p=geometry_contract()
report=analyse(
    Path("outputs/m1b3_corridor_dev/blocked_source"),
    Path("outputs/m1_current_executor_swap_smoke/source_side_grasp"),
    Path("outputs/m1b2_same_goal_dev/source"),
)
b3=report["cases"]["b3_original_topdown"]
side=report["cases"]["b3_side_grasp"]
old=report["cases"]["b2_success_reference"]
assert b3["actual_grasping_step_count"]==0 and side["actual_grasping_step_count"]==0
assert b3["minimum_tcp_cube_center_distance_m"]>.060
assert side["minimum_tcp_cube_center_distance_m"]>.100
assert old["actual_grasping_step_count"]>0
assert old["minimum_tcp_cube_center_distance_m"]<.005
assert p["goal_x_m"]==.21 and p["candidate_tip_opening_m"]>.04
assert p["fully_open_if_y_aligned_rail_intrusion_m"]>.010
assert p["candidate_if_y_aligned_rail_clearance_m"]>.002

# Verify simulator's ACTUAL loaded Panda URDF collision geometry rather
# than assuming GitHub current source matches local Conda installation.
urdf=Path(PACKAGE_ASSET_DIR)/"robots/panda/panda_v2.urdf"
if not urdf.is_file():
    raise SystemExit("M1_SOURCE_LOCAL_PANDA_V2_URDF_MISSING_NO_PHYSICS")
tree=ET.parse(urdf)
root=tree.getroot()
finger=root.find("./link[@name='panda_leftfinger']")
assert finger is not None
mount=finger.findall("collision")[0]
origin=mount.find("origin")
size=mount.find("./geometry/box")
assert origin is not None and size is not None
offset=[float(x) for x in origin.attrib["xyz"].split()]
extent=[float(x) for x in size.attrib["size"].split()]
assert abs(offset[1]-.0185)<.00001
assert abs(extent[1]-.015)<.00001
assert abs(offset[2]-.011)<.00001
assert abs(extent[2]-.020)<.00001
tcp=root.find("./joint[@name='panda_hand_tcp_joint']/origin")
assert tcp is not None
assert abs(float(tcp.attrib["xyz"].split()[2])-.1034)<.00001

# PhysX target v6 is a genuine frozen 66-step success, not re-executed.
target=Path("outputs/m1_original_scene_cartesian_pusher_v6/target_cartesian_contact_fixed")
a=target/"status.json";b=target/"physical_trace.json"
if not a.is_file() or not b.is_file():
    raise SystemExit("M1_SOURCE_V6_TARGET_REAL_SUCCESS_MISSING_NO_PHYSICS")
ts=json.loads(a.read_text());tr=json.loads(b.read_text())
assert ts.get("success") is True and ts.get("steps")==len(tr)==66
assert abs(ts["goal_error_m"]-.008193275891968195)<.0001
assert ts.get("clearance_valid") is True
assert ts["push_only"]["ever_grasped_task"] is False
assert ts["push_only"]["ever_grasped_blocker"] is False

print("M1_SOURCE_GEOMETRY_REAL_EVIDENCE_GATE",json.dumps({
    "b3_topdown_closest_tcp_m":b3["minimum_tcp_cube_center_distance_m"],
    "b3_side_closest_tcp_m":side["minimum_tcp_cube_center_distance_m"],
    "b2_different_goal_contact_reference_m":old["minimum_tcp_cube_center_distance_m"],
    "original_scene_rail_half_free_m":p["rail_inner_y_m"],
    "local_panda_urdf":str(urdf),
    "local_panda_urdf_sha256":hashlib.sha256(urdf.read_bytes()).hexdigest(),
    "full_open_mount_intrusion_y_m":p["fully_open_if_y_aligned_rail_intrusion_m"],
    "candidate_finger_opening_m":p["candidate_tip_opening_m"],
    "candidate_nominal_rail_margin_m":p["candidate_if_y_aligned_rail_clearance_m"],
    "candidate_topdown_closing_world_x":p["closing_axis_parallel_corridor_longitudinal_x"],
    "frozen_physically_successful_target_v6":True,
    "new_source_physics_runs_so_far":0,
}),flush=True)
PY

if [ -s "$case/status.json" ]; then
    echo "M1_SOURCE_ALIGNED_EXISTING_PHYSICS_PRESERVED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_SOURCE_ALIGNED_PARTIAL_RAW_PRESERVED_NO_OVERWRITE" >&2
    exit 5
else
    echo "M1_SOURCE_ALIGNED_ONE_BOUNDED_ORIGINAL_SCENE_PHYSX"
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1_source_geom_aligned_topdown_v1 \
      --out "$out" | tee "$out/source_aligned_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_SOURCE_ALIGNED_NO_STATUS_OR_PARTIAL_PHYSICS")
r=json.loads(p.read_text())
print("M1_SOURCE_GEOMETRY_ALIGNED_FINAL_GATE",json.dumps({
    "success":r.get("success"),
    "actual_steps":r.get("actual_steps"),
    "real_bilateral_grasp":r.get("actual_contact_confirmed"),
    "real_grasp_steps":r.get("actual_grasp_steps"),
    "lift_m":r.get("max_cube_lift_m"),
    "physically_over_throat":r.get("over_original_throat_physically"),
    "final_cube_xyz_m":r.get("cube_final_xyz_m"),
    "goal_error_m":r.get("goal_error_m"),
    "blocker_shift_xy_m":r.get("blocker_net_xy_m"),
    "phases":r.get("phases"),
    "actual_force_logged_steps":r.get("wall_force_measured_steps"),
    "error":r.get("error"),
    "reference_target_v6_frozen":True,
    "new_method_or_equal_info_tamp":False,
}),flush=True)
PY
