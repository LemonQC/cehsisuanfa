#!/usr/bin/env bash
# Publish only SMALL, derived, shareable science/audit summaries.
# Never stage original RoboReel videos, dataset trajectories or model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
S=outputs/v120_baselines
D=research_reports/2026-11-09_v120
mkdir -p "$D"
python - "$S" "$D" <<'PY'
import json,sys
from pathlib import Path
s,d=map(Path,sys.argv[1:])
def safe(p):
    try:return json.loads(p.read_text())
    except Exception:return None
def save(name,obj):
    (d/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
m=safe(s/"robot_data_manifest.json")
if m is not None:
    save("robot_data_manifest.json",{
       k:m.get(k) for k in ("task","status","bytes","budget_mb","candidate_file_count",
                            "error","endpoint","next_step")} | {
       "selected_demo_ids":[x.get("demo") for x in m.get("demos",[])],
       "downloaded_demo_ids":[x.get("demo") for x in m.get("downloaded",[])]})
a=safe(s/"real_robot_h5_audit.json")
if a is not None:
    rows=[]
    for x in a.get("rows",[]):
        rows.append({k:x.get(k) for k in (
         "demo","status","action_shape","articulation_name","articulation_shape",
         "qpos_9d_is_finite","action_min_preview","action_max_preview",
         "real_robot_actions","camera_videos","error")})
    save("real_robot_h5_audit.json",{
      "status":a.get("status"),"valid_demos":a.get("valid_demos",0),
      "contains_train_only":a.get("contains_train_only"),"rows":rows})
r=safe(s/"official_vid2robot_assets.json")
if r is not None:
    def shrink(section):
        x=r.get(section) or {}
        return {k:x.get(k) for k in ("status","bytes","budget_gb","budget_mb",
         "error","found","expected_videos")} | {
           "files_requested":[x2.get("path") for x2 in x.get("files",[])][:30],
           "files_downloaded":[x2.get("path") for x2 in x.get("downloaded",[])][:30]}
    save("official_vid2robot_assets.json",{
        "status":r.get("status"),
        "source_model_repo":r.get("source_model_repo"),
        "source_eval_dataset":r.get("source_eval_dataset"),
        "official_eval_ids":r.get("official_eval_ids"),
        "model":shrink("model"),"eval_videos":shrink("eval_videos"),
        "official_checkpoint_loaded":r.get("official_checkpoint_loaded"),
        "official_robot_rollout_executed":r.get("official_robot_rollout_executed")})
smoke=safe(s/"official_vid2robot_smoke/smoke_result.json")
if smoke is not None:
    # Smoke report only includes status, not raw source video/weights.
    save("official_vid2robot_smoke.json",{
       k:smoke.get(k) for k in ("type","status","checkpoint_is_official",
          "task","human_demo_id","gpu","privileged_state_in_policy",
          "checkpoint_loaded","rollout_executed","official_success",
          "official_step_record","max_rollout_steps","error",
          "is_formal_paper_score","missing")})
summary=Path(s/"official_vid2robot_smoke/official_runner_output/summary.json")
v=safe(summary)
if v is not None:
    # The published original evaluator's output for the DEV-ONLY 0080 smoke.
    (d/"official_runner_dev_summary.json").write_text(json.dumps(v,indent=2)[:300_000])
PY
cat > "$D/README.md" <<'EOF'
# v1.2 — First REAL robot train labels + official released policy baseline

- v1.1 DINOv2 ViT pilot ran correctly on RTX4090, but mean held-out
  pixel error (Full 9.73) is still above constant velocity (7.58);
  the test comprises just ONE independent video. Do NOT terminate the T-RO
  study based on this diagnostic and do not call its 2D HSV centers 3D poses.
- Official model repository (fixed):
  https://github.com/weiweigu1998/roboreel_models
  commit 79a0fb0099e73c5560114753a8706276f93aefc2.
- **Corrected upstream raw robot dataset path**:
  robot_trajectories/no_distraction/push_cube/
  (not the authors' internal alias without_distraction).
- Robot-action files are original RoboReel train H5 + base/hand camera MP4.
  Validated actions are NOT learned-policy success.
- Checkpoint is original released Vid2Robot ND, not our approximation.
- Any 0080 physics smoke uses the authors' original simulator and original
  Vid2Robot inference code, limited to exactly one preselected DEV sample.
  This is NOT a paper score. The original official last20 complete 20-episode
  protocol remains to be evaluated later, with no development on those tests.
- The official original PickCube and PushCube benchmark tasks have DIFFERENT
  objectives: do not call them a matched same-goal Pick-to-Push comparison.
- On failed downloads or dependencies, preserve BLOCKED statuses; do not
  substitute random/synthetic policies or fabricated success numbers.
- Original videos, H5 data and model checkpoints remain LOCAL ONLY.
EOF

if find "$D" -type f -size +1M | grep -q .; then
  echo "V120_AUTOPUBLISH_REJECTED: >1MB derived report";exit 2
fi
if ! git remote get-url origin 2>/dev/null | grep -Eq 'LemonQC/cehsisuanfa(\.git)?$'; then
  echo "V120_AUTOPUBLISH_REJECTED: incorrect remote";exit 2
fi
if ! git diff --cached --quiet; then
  echo "V120_AUTOPUBLISH_REJECTED: unrelated staged work";exit 2
fi
for f in "$D"/*.json "$D"/*.md; do
  if [ -f "$f" ]; then
     git add -f -- "$f"
  fi
done
if git diff --cached --quiet; then
  if ! git ls-tree -r --name-only HEAD -- "$D" | grep -q "$D/README.md"; then
    echo "V120_AUTOPUBLISH_REJECTED: no reports staged or committed";exit 2
  fi
  if git push origin HEAD:main;then
    echo "V120_AUTOPUBLISH_OK_EXISTING_COMMIT: $(git rev-parse HEAD)";exit 0
  else
    echo "V120_AUTOPUBLISH_PUSH_BLOCKED";exit 2
  fi
fi
git -c user.name="LemonQC Research Bot" \
    -c user.email="30914380+LemonQC@users.noreply.github.com" \
    commit -m "Archive v1.2 genuine robot action data and official Vid2Robot physics baseline status"
if git push origin HEAD:main;then
  echo "V120_AUTOPUBLISH_OK: $(git rev-parse HEAD)"
else
  echo "V120_AUTOPUBLISH_PUSH_BLOCKED: local report commit preserved";exit 2
fi
