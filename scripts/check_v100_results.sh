#!/usr/bin/env bash
# Inspect EXISTING v1.0 results. No training, physics, Hugging Face requests or pip.
set -u
cd "$(dirname "$0")/.."
echo "============== V1.0 SCIENCE RESULTS =============="
python - <<'PY'
import json
from pathlib import Path

root=Path("outputs/next_gate_v100")
names={
 "learnability_gate":"learnability_gate.json",
 "effect_gate":"effect_gate.json",
 "train_data":"train_data/train_discovery.json",
 "train_video_audit":"train_video_audit.json",
}
for label,rel in names.items():
    p=root/rel
    if not p.is_file():
        print(label.upper()+": MISSING_LOCAL_FILE",p)
        continue
    try:
        d=json.loads(p.read_text())
    except Exception as e:
        print(label.upper()+": INVALID_JSON",str(e));continue
    print("\n"+label.upper()+":")
    if label=="learnability_gate":
        print("  validity:",d.get("validity"))
        print("  learned_full_under_persistence:",d.get("learned_full_under_persistence"))
        for x in d.get("baseline_comparison",[]):
            print("  mode={mode} windows={windows} full_iou={mean_full_iou:.6f} end_iou={mean_end_iou:.6f}".format(
                mode=x.get("mode","?"),windows=x.get("windows","?"),
                mean_full_iou=float(x.get("mean_full_iou") or 0),
                mean_end_iou=float(x.get("mean_end_iou") or 0)))
    elif label=="effect_gate":
        print("  type:",d.get("type"))
        print("  candidate_status:",d.get("candidate_status"))
        print("  identical_to_waypoint_oracle:",d.get("identical_to_waypoint_oracle"))
        print("  source_mode:",d.get("source_mode"),"(unknown does NOT imply push)")
        print("  geometry_projected_exists:",d.get("proposed_geometric_candidate",{}).get("exists"))
    elif label=="train_data":
        print("  status:",d.get("status"))
        print("  human_training_front_videos:",d.get("human_training_front_videos"))
        print("  train_videos_downloaded:",len((d.get("downloads") or {}).get("downloaded") or []))
        print("  error:",str(d.get("error") or "")[:350])
    else:
        print("  status:",d.get("status"))
        print("  videos:",d.get("videos"))
        print("  manual_review_required:",bool(d.get("items")))
print("\nGPU_INFO: v1.0 runner has NO torch training; these metrics are CPU/network probes.")
PY
echo "============== GITHUB UPLOAD STATUS =============="
echo "LOCAL_HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
echo "LATEST_LOCAL_COMMIT: $(git log -1 --format='%h %s' 2>/dev/null || echo UNKNOWN)"
echo "REPORT_FILES_TRACKED_IN_HEAD:"
git ls-tree -r --name-only HEAD -- research_reports/2026-11-09_v100 2>/dev/null || true
echo "PUBLISH_LOG_MARKERS:"
if [ -f next_gate_v100.log ]; then
  grep -E 'AUTOPUBLISH_|SMALL_REPORT_|V100_GATE_COMPLETE|TRAIN_DATA_GATE|TRAIN_DISCOVERY_TIMEOUT' next_gate_v100.log | tail -25 || true
else
  echo "No local next_gate_v100.log"
fi
echo "CHECKING_REMOTE_HEAD (network timeout 12 seconds)..."
if command -v timeout >/dev/null 2>&1; then
  REMOTE=$(timeout 12s git ls-remote origin refs/heads/main 2>/dev/null | awk '{print $1}') || REMOTE=""
else
  REMOTE=""
fi
LOCAL=$(git rev-parse HEAD 2>/dev/null) || LOCAL=""
if [ -z "$REMOTE" ]; then
  echo "REMOTE_VERIFY_UNAVAILABLE: cannot verify remote commit on current network."
elif [ "$LOCAL" = "$REMOTE" ]; then
  echo "REMOTE_HEAD_MATCHES_LOCAL: $REMOTE"
else
  echo "REMOTE_HEAD_DIFFERS: remote=$REMOTE local=$LOCAL"
fi
echo "================================================="
