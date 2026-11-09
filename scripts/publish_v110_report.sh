#!/usr/bin/env bash
# Upload ONLY bounded, machine-readable science results to the user's
# existing GitHub repository. Never upload source human videos, robot H5,
# model checkpoint weights, HF caches or simulator binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
S=outputs/v110_baselines
D=research_reports/2026-11-09_v110
if [ ! -s "$S/official_baseline_audit.json" ]; then
  echo "AUTOPUBLISH_SKIPPED: official audit absent";exit 2
fi
mkdir -p "$D"
cp "$S/official_baseline_audit.json" "$D/official_baseline_audit.json"
if [ -s "$S/robot_data_manifest.json" ]; then
  python - "$S/robot_data_manifest.json" "$D/robot_data_manifest.json" <<'PY'
import sys,json
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
short={k:d.get(k) for k in ("task","status","endpoint","bytes","budget_mb",
 "candidate_file_count","error","next_step")}
short["demo_ids"]=[x.get("demo") for x in d.get("demos",[])]
short["downloaded_ids"]=[x.get("demo") for x in d.get("downloaded",[])]
Path(sys.argv[2]).write_text(json.dumps(short,indent=2))
PY
fi
if [ -s "$S/human_pose_annotation_audit.json" ]; then
  python - "$S/human_pose_annotation_audit.json" "$D/human_pose_annotation_audit.json" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
short={k:d.get(k) for k in ("type","status","source","tasks","endpoint","warning","error")}
short["candidate_paths"]=[x.get("path") for x in d.get("possible_annotation_files",[])][:40]
short["folder_status"]=[{"folder":f.get("folder"),"status":f.get("status"),
 "files":len(f.get("entries",[]))} for f in d.get("folders",[])]
Path(sys.argv[2]).write_text(json.dumps(short,indent=2))
PY
fi
if [ -s "$S/vit_pilot/report.json" ]; then
  cp "$S/vit_pilot/report.json" "$D/vit_report.json"
  cp "$S/vit_pilot/summary.csv" "$D/vit_summary.csv"
  cp "$S/vit_pilot/split.json" "$D/vit_split.json"
elif [ -s "$S/vit_pilot/status.json" ]; then
  cp "$S/vit_pilot/status.json" "$D/vit_status.json"
fi
cat > "$D/README.md" <<'EOF'
# v1.1 Baseline-first / pretrained ViT report

- Official baseline model code: https://github.com/weiweigu1998/roboreel_models
- Frozen reference commit: 79a0fb0099e73c5560114753a8706276f93aefc2
- Methods: video-conditioned pi0.5 uniform + keyframe, Vid2Robot, UniSkill, VLM+pi0.5.
- An official source entrypoint is NOT an executed benchmark baseline; check
  checkpoint_loaded and official_rollout_executed fields before any claim.
- ViT report is pretrained DINOv2-S ViT on real RoboReel TRAIN videos with
  video-level split, camera-plane 2D pseudo-centers, and a temporal Transformer.
- ViT diagnostic is NOT a learned control policy, a 3D effect extractor,
  a semantic segmentation QA pass or an official RoboReel success rate.
- Large videos, action H5s, model checkpoints and private env files stay on host.
- The T-RO hypothesis is NOT rejected by the earlier weak CNN pilot.
EOF
if find "$D" -type f -size +1M | grep -q .; then
  echo "AUTOPUBLISH_SKIPPED: report >1MB";exit 2
fi
if ! git remote get-url origin 2>/dev/null | grep -Eq 'LemonQC/cehsisuanfa(\.git)?$'; then
  echo "AUTOPUBLISH_SKIPPED: wrong GitHub remote";exit 2
fi
if ! git diff --cached --quiet; then
  echo "AUTOPUBLISH_SKIPPED: other changes already staged";exit 2
fi
for f in "$D"/*.json "$D"/*.csv "$D"/*.md; do
  [ -f "$f" ] && git add -f -- "$f"
done
if ! git diff --cached --quiet; then
  git -c user.name="LemonQC Research Bot" \
      -c user.email="30914380+LemonQC@users.noreply.github.com" \
      commit -m "Record v1.1 official baseline and pretrained ViT real-data results"
fi
if git push origin HEAD:main; then
  echo "AUTOPUBLISH_OK: $(git rev-parse HEAD)"
  echo "AUTOPUBLISH_URL: https://github.com/LemonQC/cehsisuanfa/tree/main/$D"
else
  echo "AUTOPUBLISH_PUSH_BLOCKED: report commit retained locally";exit 2
fi
