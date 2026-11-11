#!/usr/bin/env bash
# Current M1-B3 geometry ONLY. One preregistered v3 target safe approach.
# v2 closed-tool blocker clear remains identical; add measured no-reverse
# guard around overhead precontact + controlled forward push stages.
# No M0/B1/B2/B3/B4 prior episode reruns, no source new experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1_current_target_safe_approach_v3"
case="$out/target_safe_approach"
mkdir -p "$out"

python -m unittest discover -s tests \
  -p 'test_m1_current_target_safe_approach_v3.py' -q
python -m agcofl.m1_current_target_safe_approach_v3 --preflight

# Read-only audit of actual previous 450-step source. This validates
# the frozen result rather than hardcoding arbitrary assumptions.
python - <<'PY'
from pathlib import Path
import json
from scripts.m1_current_closed_pusher_trace_audit import analyse
r=analyse(
    Path("outputs/m1_current_closed_pusher_v2/target_closed_mplib"),
    Path("outputs/m1b3_corridor_dev/open_direct"),
)
p=r["segments"]["after_clearance_attempt"]
assert r["no_new_physics"] is True
assert r["completed_clear_phase_steps"]==82
assert r["uncompleted_post_clear_attempt_steps"]==362
assert r["observed_failure_signature"]=="CUBE_MOVED_FARTHER_FROM_GOAL_AFTER_CLEARANCE"
assert abs(p["last_cube_xyz_m"][0]+.050)<.004
assert p["cube_progress_goal_m"]<-.03
assert r["same_start_goal_as_open_direct"] is True
print("M1_TARGET_SAFE_APPROACH_REAL_EVIDENCE_GATE",json.dumps({
  "old_clear_success":True,
  "old_cube_final_x_m":p["last_cube_xyz_m"][0],
  "old_cube_net_goal_progress_m":p["cube_progress_goal_m"],
  "old_precontact_target_x_m":-.05,
  "matched_task_start_and_goal":True,
  "old_raw_input_sha256":r["input_sha256"],
  "old_data_modified":False,
  "physical_rollouts_new":0,
}),flush=True)
PY

if [ -s "$case/status.json" ]; then
    echo "M1_TARGET_SAFE_APPROACH_EXISTING_RESULT_PRESERVED $case/status.json"
elif [ -d "$case" ] && \
     find "$case" -maxdepth 1 -type f | grep -q .; then
    echo "M1_TARGET_SAFE_APPROACH_PARTIAL_RAW_FILE_NO_OVERWRITE" >&2
    exit 5
else
    echo "M1_TARGET_SAFE_APPROACH_ONE_BOUNDED_PHYSICS_TRIAL"
    timeout --signal=TERM --kill-after=10s 500s \
       python -u -m agcofl.m1_current_target_safe_approach_v3 \
         --out "$out" \
       | tee "$out/v3_console.log"
fi
python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_TARGET_SAFE_APPROACH_MISSING_REAL_STATUS")
r=json.loads(p.read_text())
print("M1_TARGET_SAFE_APPROACH_GATE",json.dumps({
 "success":r.get("success"),
 "actual_actions":r.get("action_count"),
 "closed_clearance_success":r.get("closed_tool_clearance_valid"),
 "goal_error_m":r.get("cube_goal_error_m"),
 "final_cube_xyz_m":r.get("cube_final"),
 "guard_violation":r.get("guard_violation"),
 "phase_summary":r.get("phases"),
 "error":r.get("error"),
 "source_same_goal_valid":False,
 "tamp_or_new_method_evaluated":False,
}),flush=True)
PY
