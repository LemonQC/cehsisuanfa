#!/usr/bin/env bash
# One-command *future* drawer DEV: gated scene -> preflight -> four rollouts
# Always publish a small, sanitized report, including on error. No raw media.
# Existing drawer_hf_v1 experiments MUST NOT be rerun; publish only instead.
set -Eeuo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
OUT="outputs/roboreel_video_dependency_dev/drawer_hf_v1"
STATUS="outputs/experiment_publisher/pipeline_status/drawer_hf_v1/status.json"

# Fail closed if a trial may already have written any evidence. Keep immutable.
if [[ -f "$OUT/manifest.json" || -f "$OUT/results.json" || -f "$OUT/per_trial.partial.json" || -d "$OUT/episode_00" || -d "$OUT/episode_01" ]]; then
  echo 'DRAWER_EXISTING_RUN_PRESERVED: no re-run; use publish_research_evidence --source outputs/roboreel_video_dependency_dev/drawer_hf_v1' >&2
  exit 4
fi
mkdir -p "$OUT" "$(dirname "$STATUS")"
PHASE="started"
finish() {
  local original_rc="$1"
  trap - EXIT
  set +e
  python - "$STATUS" "$PHASE" "$original_rc" <<'PY'
import json
from pathlib import Path
import sys
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'phase':sys.argv[2],'exit_code':int(sys.argv[3]),
                         'stage_result':'SUCCESS' if int(sys.argv[3])==0 else 'FAILED'},sort_keys=True)+'\n')
PY
  python -m scripts.publish_research_evidence \
    --source "$OUT" --pipeline-status "$STATUS"
  local publish_rc="$?"
  if (( original_rc != 0 )); then
    echo "DRAWER_PIPELINE_STOPPED stage=$PHASE exit=$original_rc; publisher_exit=$publish_rc" >&2
    exit "$original_rc"
  fi
  if (( publish_rc != 0 )); then
    echo "DRAWER_COMPLETED_BUT_EVIDENCE_UPLOAD_FAILED: retry publisher only" >&2
    exit "$publish_rc"
  fi
  echo 'DRAWER_RUN_AND_EVIDENCE_PUBLISHED: read experiment-evidence branch, no copy/paste needed'
}
trap 'finish "$?"' EXIT

PHASE="asset_inspect"
python -m scripts.prepare_roboreel_drawer_partnet
PHASE="scene_smoke"
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.check_roboreel_drawer_scene
PHASE="preflight"
python - <<'PY'
from pathlib import Path
from scripts.roboreel_video_dependency_dev import preflight,require_plan
p=preflight(Path.cwd(),None,'toys_in_drawer',None)
print('DRAWER_AUTO_PUBLISH_PREFLIGHT',{k:p.get(k) for k in
     ('status','reason','train_demos_usable','usable_train_demo_ids')},flush=True)
require_plan(p,2,800)
PY
PHASE="preflight_snapshot"
python -m scripts.roboreel_video_dependency_dev \
  --task toys_in_drawer --out "$OUT"
PHASE="real_rollouts"
env -u DISPLAY -u WAYLAND_DISPLAY python -u -m scripts.roboreel_video_dependency_dev \
  --task toys_in_drawer --run --episodes 2 --max-steps 800 --out "$OUT"
PHASE="complete"
