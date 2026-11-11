#!/usr/bin/env bash
# Evaluate ONLY the already-written original-MPLib target executor in
# existing M1-B3 corridor. Source failure must not block independent
# target-controller feasibility; this is NOT paired M1 evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
root="outputs/m1_current_executor_swap_smoke"
case="$root/target_official_push"
mkdir -p "$root"

python -m unittest discover -s tests -p 'test_m1_current_executor_swap.py' -q
python -m agcofl.m1_current_executor_swap --preflight

if [ -s "$case/status.json" ]; then
  echo "M1_TARGET_OFFICIAL_MPLIB_RESULT_REUSED $case/status.json"
elif [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
  echo "M1_TARGET_PARTIAL_REAL_FILES_PRESERVED_INSPECT_MANUALLY" >&2
  exit 5
else
  echo "M1_TARGET_OFFICIAL_MPLIB_ONLY_ONE_BOUNDED_PHYSX_ATTEMPT"
  timeout --signal=TERM --kill-after=10s 500s \
    python -u -m agcofl.m1_current_executor_swap \
      --mode target_official_push --out "$root" \
    | tee "$root/target_official_push_console.log"
fi

python - "$case/status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    raise SystemExit("M1_TARGET_OFFICIAL_MPLIB_NO_STATUS")
r=json.loads(p.read_text())
print("M1_TARGET_OFFICIAL_MPLIB_INDEPENDENT_RESULT",json.dumps({
    "success":r.get("success"),
    "steps":r.get("steps"),
    "goal_error_m":r.get("goal_error_m"),
    "actual_grasp_steps":r.get("actual_grasp_steps"),
    "phases":r.get("phases"),
    "error":r.get("error"),
    "source_already_matched":False,
    "is_new_method":False,
}))
PY
