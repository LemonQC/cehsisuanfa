#!/usr/bin/env bash
# Only small v1.2.1 baseline status, SHA256 and peer-review-safe evidence.
# Explicitly excludes raw VIDEO, H5, HF credentials/caches and model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
S=outputs/v120_baselines
D=research_reports/2026-11-09_v121
mkdir -p "$D"
python - "$S" "$D" <<'PY'
import json,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
def load(path):
    try:return json.loads(path.read_text())
    except Exception:return None
def save(name,obj):
    (dst/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
assets=load(src/"official_vid2robot_assets.json")
if assets:
    report={k:assets.get(k) for k in (
         "status","source_model_repo","source_eval_dataset","endpoint",
         "model_prefix","eval_prefix","official_checkpoint_loaded",
         "official_robot_rollout_executed")}
    for key in ("model","eval_videos"):
        x=assets.get(key) or {}
        report[key]={k:x.get(k) for k in (
             "status","bytes","budget_mb","budget_gb","error")}
        report[key]["verified_downloads"]=[{k:r.get(k) for k in (
             "path","bytes","sha256","transport")} for r in x.get("downloaded",[])]
        report[key]["failed_files"]=[r.get("path") for r in x.get("failures",[])]
    save("official_vid2robot_assets.json",report)
smoke=load(src/"official_vid2robot_smoke/smoke_result.json")
if smoke:
    save("official_vid2robot_smoke.json",{
       k:smoke.get(k) for k in ("type","status","checkpoint_is_official",
       "task","human_demo_id","gpu","privileged_state_in_policy",
       "checkpoint_loaded","official_state_dict_strict_loaded",
       "timm_pretrained_initialization_skipped","initializer_only_adaptation",
       "rollout_executed","official_success",
       "official_step_record","max_rollout_steps","error",
       "is_formal_paper_score","missing")})
orig=load(src/"official_vid2robot_smoke/official_runner_output/summary.json")
if orig:
    save("official_runner_dev_summary.json",orig)
PY
cat > "$D/README.md" <<'EOF'
# v1.2.1 — Released Vid2Robot checkpoint and official evaluator recovery

v1.2 already obtained and validated **two authentic TRAIN RoboReel robot
H5 datasets**, with 138x8 and 136x8 policy actions and real base/hand camera
MP4s. Those files stay local and are NOT downloaded again in this stage.

Previous v1.2 blocker: HF metadata detected a ~2.37GB original released
Vid2Robot checkpoint and 20 original evaluation videos (~97.45MB). The
original HF download attempt failed with LocalEntryNotFoundError. This stage:
- explicitly binds hf_hub_download to the official or mirror endpoint;
- reuses any local files that match the PUBLISHED exact byte size;
- reuses the earlier official EVAL video 0080 only if exact size matches;
- retries and resumes HTTP downloads from verified HF resolve hosts;
- writes per-file verified local SHA256 without sharing original bytes;
- runs only the authors' original Vid2Robot/M aniSkill runner, never a
  substituted policy or privileged-state scripted controller.
- DEV-only single heldout 0080 rollout is a pipeline check, NOT a paper
  score; no tuning/model selection is allowed on EVAL demonstrations.

No official success can be claimed unless the JSON says
rollout_executed=true and a measured official_success flag is present.
Any contact or robot dynamics result requires actual physics rollout.
EOF
# Correct a nontechnical README typo while retaining stable facts.
sed -i 's@Vid2Robot/M aniSkill@Vid2Robot/ManiSkill@' "$D/README.md"
if find "$D" -type f -size +1M | grep -q .; then
 echo "V121_AUTOPUBLISH_REJECTED: file above 1MB";exit 2
fi
if ! git remote get-url origin 2>/dev/null | grep -Eq 'LemonQC/cehsisuanfa(\.git)?$'; then
 echo "V121_AUTOPUBLISH_REJECTED: wrong git remote";exit 2
fi
if ! git diff --cached --quiet; then
 echo "V121_AUTOPUBLISH_REJECTED: unrelated staged files";exit 2
fi
for f in "$D"/*.json "$D"/*.md; do
 [ -f "$f" ] && git add -f -- "$f"
done
if ! git diff --cached --quiet; then
 git -c user.name="LemonQC Research Bot" \
     -c user.email="30914380+LemonQC@users.noreply.github.com" \
     commit -m "Record v1.2.1 official Vid2Robot checkpoint recovery and physics rollout status"
fi
if git push origin HEAD:main; then
 echo "V121_AUTOPUBLISH_OK: $(git rev-parse HEAD)"
 echo "V121_AUTOPUBLISH_URL: https://github.com/LemonQC/cehsisuanfa/tree/main/$D"
else
 echo "V121_AUTOPUBLISH_PUSH_BLOCKED: local commit retained";exit 2
fi
