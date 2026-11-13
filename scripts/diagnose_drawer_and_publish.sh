#!/usr/bin/env bash
# One-time, read-only frozen ObjectInDrawer action diagnosis; publish automatically.
# NEVER imports ManiSkill/OpenPI, invokes env.step, downloads assets or changes the source.
set -Eeuo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

SRC="outputs/roboreel_video_dependency_dev/drawer_hf_v1"
OUT="outputs/roboreel_video_dependency_diagnosis/drawer_hf_v1"

if [[ ! -f "$SRC/results.json" ]]; then
  echo "DRAWER_DIAG_BLOCKED: original frozen results.json missing" >&2
  exit 3
fi

if [[ -e "$OUT" ]]; then
  if [[ -f "$OUT/diagnosis.json" && -f "$OUT/SUMMARY.md" ]]; then
    echo "DRAWER_EXISTING_DIAGNOSIS_PRESERVED: publish only"
  else
    echo "DRAWER_DIAG_BLOCKED: partial output exists; do not overwrite" >&2
    exit 4
  fi
else
  python -m scripts.diagnose_drawer_video_dev --root "$SRC" --out "$OUT"
fi

# Never publish human video, PartNet, weights, raw actions, or full MP4s.
# Publisher sanitizes action metrics; robot generated montage only.
python -u -m scripts.publish_research_evidence --source "$OUT"
echo "DRAWER_DIAG_PUBLISHED: review tro1-experiment-evidence/evidence/runs; no GPU rerun"
