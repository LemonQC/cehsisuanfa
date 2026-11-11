#!/usr/bin/env bash
# M1B4 — ONE VERSIONED PHYSICAL RESCUE, unchanged M1B3 scene.
# Must stop (no simulation) unless prior source truly grasped/lifted/carried.
# Never reruns old M0/M1B1/M1B2/M1B3 episodes.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1b4_one_time_feasibility"
mkdir -p "$out"

python -m unittest discover -s tests -p 'test_m1b4_one_time_feasibility.py' -q
echo "M1B4_UNIT_TESTS_PASSED_NOW_READING_REAL_FROZEN_M1B3_DIAGNOSIS"

# This is the FIRST genuine user-local evidence check. Unit tests only
# use temporary synthetic fixtures whose NO_GO marker is not displayed.
# Exit 6 means NO-GO and MUST stop before any PhysX execution.
python -m agcofl.m1b4_one_time_feasibility \
  --old-root outputs/m1b3_corridor_dev \
  --require-physical-source-evidence \
  | tee "$out/frozen_evidence_preflight.log"

for mode in source_lower_fallback target_align_after_clear; do
  case="$out/$mode"
  if [ -s "$case/status.json" ]; then
    echo "M1B4_REUSE_IMMUTABLE_RESULT $mode"
  else
    if [ -d "$case" ] && find "$case" -maxdepth 1 -type f | grep -q .; then
      echo "M1B4_PARTIAL_CASE_NEEDS_MANUAL_REVIEW $case" >&2
      exit 5
    fi
    echo "M1B4_EXECUTE_ONCE $mode seed=120"
    if command -v timeout >/dev/null 2>&1; then
      timeout --signal=TERM --kill-after=10s 500s \
        python -u -m agcofl.m1b4_one_time_feasibility \
        --mode "$mode" --out "$out" \
      | tee "$out/$mode-console.log"
    else
      python -u -m agcofl.m1b4_one_time_feasibility \
        --mode "$mode" --out "$out" \
      | tee "$out/$mode-console.log"
    fi
  fi

  if [ "$mode" = source_lower_fallback ]; then
    # A source failure means the required matched-goal pair is absent;
    # abort before running a target-only episode that cannot change that.
    if ! python -c \
      "import json; from pathlib import Path; p=Path('$case/status.json'); assert json.loads(p.read_text())['gate']['valid'] is True, 'M1B4_NO_GO_SOURCE_NOT_VALID';"; then
      echo "M1B4_STOP_SOURCE_FAILED_NO_FURTHER_PHYSICS" >&2
      exit 7
    fi
  fi
done

python -m scripts.m1b4_validate_once --out "$out"
echo "M1B4_FEASIBILITY_LAST_ALLOWED_ATTEMPT_COMPLETE_NOT_TAMP_NOT_METHOD"
