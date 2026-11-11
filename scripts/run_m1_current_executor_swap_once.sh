#!/usr/bin/env bash
# Replace execution modules in ORIGINAL M1B3 geometry, ONE scoped smoke.
# Does NOT modify old H5/videos, change task/goals, or train a new method.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1_current_executor_swap_smoke"
mkdir -p "$out"

python -m unittest discover -s tests -p 'test_m1_current_executor_swap.py' -q
python -m agcofl.m1_current_executor_swap --preflight

source_status="$out/source_side_grasp/status.json"
if [ -s "$source_status" ]; then
  echo "M1_EXECUTOR_SOURCE_RESULT_REUSED $source_status"
else
  if [ -d "$out/source_side_grasp" ] && \
     find "$out/source_side_grasp" -maxdepth 1 -type f | grep -q .; then
    echo "PARTIAL_SOURCE_EVIDENCE_NO_OVERWRITE" >&2
    exit 5
  fi
  echo "M1_EXECUTOR_SOURCE_OFFICIAL_SIDE_GRASP_SINGLE_SMOKE"
  timeout --signal=TERM --kill-after=10s 500s \
    python -u -m agcofl.m1_current_executor_swap \
      --mode source_side_grasp --out "$out" \
    | tee "$out/source-console.log"
fi

# A physically successful source grasp and matched same-scene final
# cube goal are REQUIRED. Planner exit zero is not evidence.
python - "$source_status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("M1_EXECUTOR_SOURCE_MISSING_NO_TARGET_RUN")
r=json.loads(p.read_text())
print("M1_EXECUTOR_REAL_SOURCE_GATE",json.dumps({
  "success":r.get("success"),"actual_grasp_steps":r.get("actual_grasp_steps"),
  "observed_lift_m":r.get("observed_lift_m"),"error":r.get("error"),
  "goal_error_m":r.get("goal_error_m")
}))
if (r.get("version")!="m1_current_task_official_executor_swap_v1"
    or r.get("success") is not True
    or r.get("actual_grasp_steps",0)<=0
    or r.get("goal_error_m",1)>=.020):
    raise SystemExit("M1_EXECUTOR_SOURCE_NO_GO_NO_TARGET_RUN")
PY

target_status="$out/target_official_push/status.json"
if [ -s "$target_status" ]; then
  echo "M1_EXECUTOR_TARGET_RESULT_REUSED $target_status"
else
  if [ -d "$out/target_official_push" ] && \
     find "$out/target_official_push" -maxdepth 1 -type f | grep -q .; then
    echo "PARTIAL_TARGET_EVIDENCE_NO_OVERWRITE" >&2
    exit 5
  fi
  echo "M1_EXECUTOR_TARGET_UPSTREAM_PUSHCUBE_MPLIB_SINGLE_SMOKE"
  timeout --signal=TERM --kill-after=10s 500s \
    python -u -m agcofl.m1_current_executor_swap \
      --mode target_official_push --out "$out" \
    | tee "$out/target-console.log"
fi
echo "M1_EXECUTOR_SWAP_CASES_FINISHED_VERIFY_REAL_H5_TRACES_NOT_A_METHOD"
