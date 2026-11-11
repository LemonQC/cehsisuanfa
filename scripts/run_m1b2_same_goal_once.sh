#!/usr/bin/env bash
# M1-B2: exactly one bounded DEV scene/seed, three physical cases.
# Runs with original ManiSkill solver and existing Push teacher; never trains.
# NO checkpoint downloads, no M0 rerun, no hidden seeds/threshold tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
out="outputs/m1b2_same_goal_dev"
mkdir -p "$out"

python -m unittest discover -s tests -p 'test_m1b2_same_goal_effect_physics.py' -q

for mode in source target_direct target_clear; do
  status="$out/$mode/status.json"
  if [ -s "$status" ]; then
    echo "M1B2_PRESERVE_COMPLETED_CASE $mode $status"
    continue
  fi
  if [ -f "$out/$mode/$mode.h5" ] || [ -f "$out/$mode/$mode.json" ]; then
    echo "M1B2_PARTIAL_RAW_EXISTS_NEEDS_MANUAL_REVIEW $mode" >&2
    exit 5
  fi
  echo "M1B2_EXECUTE_FIXED_CASE $mode seed=120"
  # Bound MPLib planning time. Exit cleanly on timeout rather than
  # launching unbounded expert or changing geometry for this seed.
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1b2_same_goal_effect_physics \
       --mode "$mode" --seed 120 --max-steps 450 --out "$out" \
      | tee "$out/$mode-console.log"
  else
    python -u -m agcofl.m1b2_same_goal_effect_physics \
       --mode "$mode" --seed 120 --max-steps 450 --out "$out" \
      | tee "$out/$mode-console.log"
  fi
done

python -m scripts.m1b2_validate_paired_physics --root "$out"
echo "M1B2_DEV_FIXED_PILOT_COMPLETE_NOT_A_PAPER_RESULT"
echo "RAW: $out/{source,target_direct,target_clear}/{status.json,physical_trace.json,*.mp4,*.h5}"
