#!/usr/bin/env bash
# One immutable 4-case M1-B3 DEV pilot. NEVER touches old M1B2 output.
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

out="outputs/m1b3_corridor_dev"
mkdir -p "$out"

# Cheap fully independent standard-library/Python-Numpy preflight.
python -m unittest discover -s tests -p 'test_m1b3_corridor_physics.py' -q
python -m agcofl.m1b3_corridor_physics --preflight-only \
  | tee "$out/geometry_preflight.log"

for mode in blocked_source blocked_direct blocked_clear open_direct; do
  status="$out/$mode/status.json"
  raw="$out/$mode/$mode.h5"
  meta="$out/$mode/$mode.json"
  if [ -s "$status" ]; then
    echo "M1B3_PRESERVE_IMMUTABLE_CASE $mode"
    continue
  fi
  if [ -e "$raw" ] || [ -e "$meta" ]; then
    echo "M1B3_PARTIAL_RAW_EVIDENCE_DO_NOT_OVERWRITE $mode" >&2
    exit 5
  fi
  echo "M1B3_EXECUTE_FIXED_PHYSICS_CASE $mode seed=120"
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10s 500s \
      python -u -m agcofl.m1b3_corridor_physics \
      --mode "$mode" --seed 120 --max-steps 450 --out "$out" \
      | tee "$out/$mode-console.log"
  else
    python -u -m agcofl.m1b3_corridor_physics \
      --mode "$mode" --seed 120 --max-steps 450 --out "$out" \
      | tee "$out/$mode-console.log"
  fi
done
python -m scripts.m1b3_validate_corridor --out "$out"
echo "M1B3_DEV_PHYSICS_FOUR_CASES_RECORDED_NOT_A_TAMP_BASELINE"
echo "All results are local under: $out"
