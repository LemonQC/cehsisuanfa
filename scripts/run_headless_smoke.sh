#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
python -u -X faulthandler -m agcofl.cli doctor 2>&1 | tee outputs_doctor.log
