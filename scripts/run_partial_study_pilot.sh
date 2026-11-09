#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

python -m pytest -q tests/test_partial_flow_study.py
python -m agcofl.cli test-loss

# CODE-ONLY test: NO real human videos or robot outcomes.
python -m agcofl.partial_flow_study make-proxy \
  --out outputs/proxy_supervision_v070 --layouts 12 --seed 42

python -u -m agcofl.partial_flow_study train \
  --manifest outputs/proxy_supervision_v070/manifest.jsonl \
  --out outputs/proxy_full_anchor_end_v070 \
  --seed 42 --epochs 15 --human-weight 0.5 \
  --device auto --allow-proxy

python -m agcofl.plot_partial_study --out outputs/proxy_full_anchor_end_v070

echo "PROXY_DIAGNOSTICS_ONLY:"
cat outputs/proxy_full_anchor_end_v070/report.json
echo "CSV: outputs/proxy_full_anchor_end_v070/robot_test_summary.csv"
echo "CSV: outputs/proxy_full_anchor_end_v070/paired_differences.csv"
echo "PNG: outputs/proxy_full_anchor_end_v070/offline_comparison.png"
