#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
python -u -m agcofl.cli gate-a --ckpt outputs/models/flow_bc_ee_v030.pt --seed 9000 --episodes 5 --max-steps 500 --video
