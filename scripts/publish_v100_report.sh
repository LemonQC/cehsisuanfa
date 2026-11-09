#!/usr/bin/env bash
# Push ONLY small derived experimental reports. No videos, ZIPs, weights.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=research_reports/2026-11-09_v100
SOURCE=outputs/next_gate_v100
mkdir -p "$DIR"
for f in learnability_gate.json learning_vs_motion_baselines.csv effect_gate.json train_video_audit.json; do
  if [ -f "$SOURCE/$f" ]; then cp "$SOURCE/$f" "$DIR/$f"; fi
done
if [ -f "$SOURCE/train_data/train_discovery.json" ]; then
  # Training dataset metadata can contain thousands of file names.
  # Publish a concise sanitized summary; keep full listing LOCAL only.
  python - "$SOURCE/train_data/train_discovery.json" "$DIR/train_discovery.json" <<'PY'
import json,sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text())
subset={k:data.get(k) for k in (
    "dataset","endpoint","status","error","human_training_front_videos",
    "limit_hit","warning")}
subset["candidate_examples"]=(data.get("candidates") or [])[:20]
subset["remote_roots"]=(data.get("roots") or [])[:20]
downloads=data.get("downloads") or {}
subset["train_videos_downloaded"]=len(downloads.get("downloaded") or [])
Path(sys.argv[2]).write_text(json.dumps(subset,indent=2))
PY
fi
cat > "$DIR/README.md" <<'EOF'
# v1.0 experiment gates | 2026-11-09

- Actual v0.9 human-video predictions are compared with no-learning motion baselines.
- RoboReel evaluation demos 0080-0099 are not valid training videos for official scoring.
- The saved RoboReel PushCube episode is REAL physical simulation but not a Pick source.
- Geometric projection matching Goal+Waypoint Oracle offers no independent innovation.
- Candidate paths are not physically verified until actual robot execution.
- An unavailable China mirror is a data blocker, not a passed experiment.
EOF

if find "$DIR" -type f -size +1M | grep -q .; then
  echo "AUTOPUBLISH_SKIPPED: report over 1 MB";exit 2
fi
if find "$DIR" -type f ! -name '*.md' ! -name '*.json' ! -name '*.csv' | grep -q .; then
  echo "AUTOPUBLISH_SKIPPED: unexpected file type";exit 2
fi
if ! git remote get-url origin 2>/dev/null | grep -Eq 'LemonQC/cehsisuanfa(\.git)?$'; then
  echo "AUTOPUBLISH_SKIPPED: origin mismatch";exit 2
fi
if ! git diff --cached --quiet; then
  echo "AUTOPUBLISH_SKIPPED: other staged files";exit 2
fi
git add -- "$DIR"
if git diff --cached --quiet; then echo "AUTOPUBLISH_NO_CHANGES";exit 0;fi
if ! git -c user.name="LemonQC Research Bot" -c user.email="30914380+LemonQC@users.noreply.github.com" \
    commit -m "Archive v1.0 measured video and geometry research gates"; then
  echo "AUTOPUBLISH_COMMIT_FAILED";exit 2
fi
if git push origin HEAD:main; then
  echo "AUTOPUBLISH_OK: $(git rev-parse HEAD) https://github.com/LemonQC/cehsisuanfa/tree/main/$DIR"
else
  echo "AUTOPUBLISH_PUSH_BLOCKED: local commit retained; offline Github access may be unavailable"; exit 2
fi
