# StackCups genuine TRAIN-video paired DEV — real 4-rollout triage

Date: 2026-11-13. Project: TRO1. Source: user-supplied RTX 4090 / pi05 terminal output. **Do not imply the private server files were opened or uploaded.**

## Verified as of this message

- Official task `StackCups-v1`; pinned video-conditioned `demo_pi05` checkpoint; no PartNet-Mobility; two original human TRAIN videos `0000` and `0001`; fixed matched physical state and policy RNG per paired episode.
- Real result log: `ROBOREEL_VIDEO_DEV_RESULT {"status":"DEV_COMPLETE_NOT_FORMAL","task":"stack_cups","pairs":2,"rollouts":4,"any_pair_success_difference":false,...}`.
- Two explicitly observed trial lines:
  - `{"episode":1,"source_train_demo_id":1,"success":false,"steps":800,"error":null}`
  - `{"episode":1,"source_train_demo_id":0,"success":false,"steps":800,"error":null}`
- **Episode 0's individual success values were NOT included in the user terminal excerpt.** Do **not** assert 0/4 success, because `any_pair_success_difference=false` merely means outcomes agree *within* each episode; the first pair might both succeed or both fail.
- `os.fork()`/JAX multithreading warning occurred during subprocess video encoding; the reported 4 trial execution completed. This warning by itself is **not** evidence of physics or policy failure. Check `trials[*].video_error` before relying on videos.
- No paired success-outcome reversal across 2 scenes. This **does not mean** demonstrations are ignored; need action discrepancy metrics. Two pairs are too small for causal claims, T-RO novelty, or efficacy.
- The physical experiment has no observed success-vs-failure distinction attributable to choice of demonstration; it has not tested a new event-progress synchronisation method or strong comparators.

## Existing on-server evidence to inspect **without any new simulation**

Immutable DEV root: `outputs/roboreel_video_dependency_dev/stack_v1`.

Script `scripts/roboreel_video_dependency_dev.py` already saved:

- `results.json`: `pairs[].mean_action_l2_first_common_steps`, `pairs[].successes`, `pairs[].source_ids`, `trials[]` metadata.
- `episode_00/source_0000/actions.npy`, `episode_00/source_0001/actions.npy`, and their episode-01 peers; `success_trace.json`, `status.json`, optionally `rollout.mp4`.
- `per_trial.csv` and `SUMMARY.md`. The console truncated Episode 0, so inspect actual saved result file rather than assuming its values.

Read-only extraction command:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/roboreel_video_dependency_dev/stack_v1/results.json')
r=json.loads(p.read_text())
print('VERIFIED_SUMMARY', {'task':r['task'],'rollouts':r['real_rollouts'],
                          'paired_episodes':r['paired_episodes']})
for x in r['pairs']:
    print('PAIR', {k:x.get(k) for k in (
        'episode','source_ids','successes','success_disagree',
        'same_reset_seed','same_rng_seed','mean_action_l2_first_common_steps')})
for x in r['trials']:
    print('TRIAL', {k:x.get(k) for k in (
        'episode','source_train_demo_id','success','steps','error',
        'video_error','video_path','last_success')})
PY
```

Do **not** run `--run` again, change video source, retune policy or expand sample size at this decision point.

## Pre-registered decision gate

1. Verify Episode 0 success values; inspect pairwise 8D action-L2. If both pairs fail, or one pair succeeds without differences between videos, task completion does not show discriminative conditioning utility; no T-RO GO.
2. If action-L2 is near zero, first check whether two source embeddings and model input actually differ and whether same-scene response is insensitive. This requires raw action arrays; do not invent a numerical threshold without inspecting action scale.
3. If action-L2 is appreciable but success remains identical, inspect videos to classify failure stage (reach/grasp, lift, first stack, second stack, instability). Do not infer stage from binary success traces; saved `success_trace.json` only records final task success at each step.
4. Even successful distinct actions would not prove the human video contains *necessary* information, since both videos describe the same task and no condition demands a divergent preference.
5. Until direct evidence supports a distinct, necessary and measurable contribution, report **NO-GO for novelty claim**, not a failure of the infrastructure, and do not launch more speculative baselines.

## Evidence integrity

- The present document records only terminal observations in the conversation, not unseen `results.json` contents.
- Do not commit private raw MP4/NPY weights or videos to GitHub by default.
- Official EVAL heldout 0080–0099 unaffected; existing M0 baselines not rerun.
