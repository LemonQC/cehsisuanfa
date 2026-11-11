#!/usr/bin/env bash
# Same existing M1-B3 environment, ONE original-Panda gripper-sign control.
# Never rerun frozen M0, M1B1, M1B2, M1B3, side-grasp, or failed open MPLib.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

out="outputs/m1_current_closed_pusher_v2"
mkdir -p "$out"

python -m unittest discover -s tests \
  -p 'test_m1_current_closed_pusher_ablation.py' -q

python -m agcofl.m1_current_closed_pusher_ablation --preflight

# Validate both genuinely saved old physical conditions BEFORE this
# one-factor ablation. Old outcome labels were not gripper-open evidence.
python - <<'PY'
import json
from pathlib import Path
old_open=Path("outputs/m1_current_executor_swap_smoke/target_official_push")
old_closed=Path("outputs/m1b3_corridor_dev/blocked_clear")
def load(root):
    p=root/"status.json"
    q=root/"physical_trace.json"
    if not p.is_file() or not q.is_file():
        raise SystemExit(f"NO_REAL_PRIOR_PHYSICS_SOURCE: {root}")
    return json.loads(p.read_text()), json.loads(q.read_text())
a,at=load(old_open)
b,bt=load(old_closed)
if not (a.get("mode")=="target_official_push"
        and a.get("version")=="m1_current_task_official_executor_swap_v1"
        and a.get("success") is False and a.get("steps")==88
        and len(at)==88 and all(x["gripper_action"]>0 for x in at)
        and b.get("scene")=="AGCOFLM1B3Corridor-v0"
        and b.get("seed")==120
        and b.get("gate",{}).get("blocker_cleared_to_side_pocket") is True
        and b.get("phases",[{}])[0].get("name")=="clear_blocker"
        and b["phases"][0].get("success") is True
        and b["phases"][0].get("steps")==32
        and len(bt)>=32
        and all(x["gripper_action"]<0 for x in bt[:32])):
    raise SystemExit("NO_GO: original Panda OPEN/CLOSED action evidence not verified; no new physics")
print("M1_TARGET_SIGN_ROOT_CAUSE_EVIDENCE",json.dumps({
    "old_open_mplib_clearance_steps":a["steps"],
    "old_open_mplib_clearance_error_m":a["phases"][0]["actual_error_m"],
    "old_successful_closed_clearance_steps":b["phases"][0]["steps"],
    "closed_sign_matches_published_panda_push_cube":True,
    "old_data_changed":False,
    "source_pair_success_claimed":False,
}))
PY

case="$out/target_closed_mplib"
if [ -s "$case/status.json" ]; then
    echo "M1_TARGET_CLOSED_PUSHER_EXISTING_RESULT_PRESERVED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_TARGET_CLOSED_PUSHER_PARTIAL_RAW_PRESERVED_NO_OVERWRITE" >&2
    exit 5
else
    echo "M1_TARGET_CLOSED_PUSHER_ONE_FACTOR_PHYSX_SINGLE_TRIAL"
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1_current_closed_pusher_ablation --out "$out" \
      | tee "$out/target_closed_pusher_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_TARGET_CLOSED_PUSHER_STATUS_MISSING")
r=json.loads(p.read_text())
print("M1_TARGET_CLOSED_PUSHER_RESULT",json.dumps({
    "success":r.get("success"),
    "steps":r.get("steps"),
    "error":r.get("error"),
    "phases":r.get("phases",[]),
    "push_only_physical_check":r.get("push_only"),
    "original_goal_error_m":r.get("task_goal_error_m"),
    "source_matched":False,
    "TAMP_or_learned_policy_evaluated":False,
}))
PY
