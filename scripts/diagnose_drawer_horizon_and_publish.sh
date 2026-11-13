#!/usr/bin/env bash
# Analyze *existing* four frozen ObjectInDrawer success traces, publish safely.
# No physics, policy inference, HF downloads, or training.
set -Eeuo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SRC='outputs/roboreel_video_dependency_dev/drawer_hf_v1'
DST='outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1'
if [[ -e "$DST" ]]; then
  if [[ -f "$DST/horizon_audit.json" && -f "$DST/SUMMARY.md" ]]; then
    echo 'HORIZON_AUDIT_ALREADY_PRESENT: preserve existing bytes, upload only'
  else
    echo 'HORIZON_AUDIT_PARTIAL_OUTPUT_STOP: manual review required' >&2
    exit 4
  fi
else
  python -m scripts.audit_drawer_success_horizon --root "$SRC" --out "$DST"
fi
python -u -m scripts.publish_research_evidence --source "$DST" --no-preview
echo 'DRAWER_HORIZON_EVIDENCE_PUBLISHED: inspect tro1-experiment-evidence branch'
