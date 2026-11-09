#!/usr/bin/env bash
# Publish only original official trial outcome metadata + protocol summary.
# No model weights, original RGB videos, H5 trajectories, logs or keys.
set -euo pipefail
cd "$(dirname "$0")/.."
S=outputs/v130_official_vid2robot
D=research_reports/2026-11-09_v130
mkdir -p "$D"
python - "$S" "$D" <<'PY'
import json,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
def fetch(name):
    p=src/name
    if not p.is_file():return None
    return json.loads(p.read_text())
def save(name,v):
    (dst/name).write_text(json.dumps(v,ensure_ascii=False,indent=2))
run=fetch("run_status.json")
agg=fetch("aggregated_summary.json")
if run is None or agg is None:
    save("status.json",{"status":"V130_NOT_YET_COMPLETED",
          "detail":"Runner status or aggregate absent; not an official result"})
else:
    save("status.json",{
       k:run.get(k) for k in ("type","status","official_model","task",
           "original_source_commit","seed_offsets_requested",
           "no_eval_training_or_tuning","passes_completed","errors")})
    save("official_aggregated_summary.json",agg)
    # The per-episode outcomes in the aggregate are already the original
    # physics evaluator's 20 named examples, not generated predictions.
PY
cat > "$D/README.md" <<'EOF'
# v1.3 — Actual original Vid2Robot released policy, official RoboReel protocol

Scope: **PushCube clean human + clean environment only**; no 10-task or
WD/ED claims. Human EVAL videos 0080–0099, original physical simulator,
authors' published Vid2Robot checkpoint. Original pd_joint_pos action,
chunk_size=1, max_rollout_steps=800 and three environment seed offsets
0/100000/200000. No checkpoint tuning on official EVAL videos.

Only official_aggregated_summary.json with full_3pass_complete=true
and status.json.status=OFFICIAL_VID2ROBOT_ND_3PASS_EVALUATED permits a
three-pass performance statement. Missing or interrupted seeds stay partial.

Reproduction delta from unmodified authors' Python evaluator:
a thin launch adapter suppresses timm's redundant ImageNet initializer
download while authors' complete original Vid2Robot state_dict is strictly
loaded. No changes to architecture, learned parameters, action policy,
original RoboReel success criterion or physics.
EOF
if find "$D" -type f -size +1M | grep -q .; then
 echo "V130_AUTOPUBLISH_REJECTED: too large"; exit 2
fi
if ! git remote get-url origin 2>/dev/null | grep -Eq 'LemonQC/cehsisuanfa(\.git)?$';then
 echo "V130_AUTOPUBLISH_REJECTED: wrong remote";exit 2
fi
if ! git diff --cached --quiet;then
 echo "V130_AUTOPUBLISH_REJECTED: unrelated staged changes";exit 2
fi
for f in "$D"/*.json "$D"/*.md; do
 [ -f "$f" ] && git add -f -- "$f"
done
if ! git diff --cached --quiet;then
 git -c user.name="LemonQC Research Bot" \
   -c user.email="30914380+LemonQC@users.noreply.github.com" \
   commit -m "Archive genuine official Vid2Robot 20-episode three-seed RoboReel physics evaluation" 
fi
if git push origin HEAD:main;then
 echo "V130_AUTOPUBLISH_OK: $(git rev-parse HEAD)"
 echo "V130_AUTOPUBLISH_URL: https://github.com/LemonQC/cehsisuanfa/tree/main/$D"
else
 echo "V130_AUTOPUBLISH_PUSH_BLOCKED: committed reports preserved"; exit 2
fi
