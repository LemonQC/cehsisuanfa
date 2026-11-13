# B06 UniSkill — Real StackCups DEV Result Audit (2026-11-13)

## 1. Status and authoritative evidence

- Evidence: `tro1-experiment-evidence/evidence/runs/b06_uniskill_stackcups_dev_3c629e97bd06f52d/evidence.json`.
- Official RoboReel parent revision `79a0fb0099e73c5560114753a8706276f93aefc2`. Genuine RoboReel StackCups PHYSX environment and released UniSkill policy + UniSkill IDM.
- **DEV_COMPLETE_NOT_FORMAL**, **4/4 rollouts finished, 0/4 task successes, 3,200 actual PhysX actions, 800 steps per rollout, zero runtime errors**.
- Two TRAIN source videos (IDs 0000/0001), two matched reset seeds (310000/310001), matched torch RNG seeds in each pair (560000/560001). This is **NOT** the official held-out last20×3 evaluation and is not sufficient to conclude UniSkill's official task success.
- Evidence contains SHA256 for both checkpoints, each of the two source videos and skill arrays, per-trial action trace and MP4. No new training.

## 2. Frozen recorded measurements

| Episode | Source TRAIN ID | Steps | Success | Runtime error |
|---|---:|---:|---|---|
| 0 | 0000 | 800 | false | null |
| 0 | 0001 | 800 | false | null |
| 1 | 0001 | 800 | false | null |
| 1 | 0000 | 800 | false | null |

- Episode 0, first 20 common steps: mean action L2 difference **0.04923475533723831**.
- Episode 1, first 20 common steps: mean action L2 difference **0.07775086909532547**.
- In each pair, initial state and torch RNG seed are equal and source order is reversed across episodes.
- Action differences show that different human skill inputs induce some different actions, **not** that video semantics are correctly grounded, nor that these differences are helpful.
- Both pairs have **no success preference reversal**. This DEV cannot demonstrate video-driven behavior selection necessity.

## 3. Decision

**B06 is now physically runnable but not yet scientifically validated as a faithful benchmark comparison.** Do not mark B06 as blocked by installation, and equally do not report 0% as an official test score. Do not immediately burn GPU time on repeated same-seed runs or a formal 20×3 test before checking behavior traces.

First audit the **already recorded four robot-policy MP4s**. They are not human TRAIN/EVAL demonstrations. Inspect initial, intermediate and final frames to distinguish:
1. no approach/contact vs contact but unsuccessful grasp;
2. grasp succeeds but placement/stacking fails;
3. long idle/saturation or robot motion not consistent with the task;
4. rendering/action/normalization mismatch requiring author-code audit.

No one of these failure modes is established by GitHub numeric evidence alone.

## 4. One read-only foreground step; no new PhysX

The repo already contains `scripts/publish_research_evidence.py`, which constructs a **small 4×6 contact sheet of sampled frames from recorded ROBOT MP4s**, not any human demo. It uploads only sanitized measurements, a README, and (if decoding succeeds) a JPEG <=850 KB to the dedicated `tro1-experiment-evidence` branch. It neither modifies checkpoints nor calls simulator. Use the existing independent UniSkill Python to ensure imageio/Pillow are present:

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main

outputs/v120_baselines/conda_envs/uniskill/bin/python \
  scripts/publish_research_evidence.py \
  --source outputs/research_runs/b06_uniskill_stackcups_dev_v1 \
  --dry-run

# Check file_names includes "robot_rollout_preview.jpg"; then publish:
outputs/v120_baselines/conda_envs/uniskill/bin/python \
  scripts/publish_research_evidence.py \
  --source outputs/research_runs/b06_uniskill_stackcups_dev_v1
```

If image decoding fails, `--dry-run` will omit the preview; **do not** say video was verified. If the preview is published, inspect actual robot motion before deciding on one targeted reproduction fix or a frozen formal held-out protocol.

## 5. Scientific scope

- B06 is a one-method-at-a-time DEV baseline. B07 and any new methods remain untouched.
- Evaluation protocol remains unchanged; do not adjust success predicates, task tolerance, privileged inputs, or simulation to make the number look better.
- Preserve the immutable existing 0/4 result: any later technical repair requires a **new, clearly versioned result** plus an explicit difference from the original.

## 6. 2026-11-13 GitHub video audit: preview ABSENT; publisher schema mismatch found

Direct GitHub evidence-branch tree inspection found exactly one file under `evidence/runs/b06_uniskill_stackcups_dev_3c629e97bd06f52d/`: `evidence.json`; no `robot_rollout_preview.jpg`, no robot MP4s. The latest evidence commit for that directory only uploaded `evidence.json`. There are preview JPGs for old drawer runs, but none for B06. **Do not infer observed robot grasp/contact/placement behavior from success metrics alone.**

Root cause in the previously recommended video publisher: `scripts/publish_research_evidence.py:read_result()` expected `per_trial.partial.json` to be a bare array, whereas B06's append-only `save_progress_json()` records `{"trials": [...]} `. This triggers `PARTIAL_TRIALS_MUST_BE_JSON_ARRAY` even if the separate `results.json` is complete, so no JPEG can be built/uploaded. Fixed `read_result` to accept both the legacy array and the precise B06 `{"trials": list}` form, rejecting malformed/unrecognized objects; stable B06 read-only summaries also accept `rollouts`/`episodes` fields as well as historical `real_rollouts`/`paired_episodes`. Add offline regression tests to `tests/test_publish_research_evidence.py` with a completed 4-rollout B06 mock containing the actual progress schema.

**Next action requires only existing ROBOT MP4s (no PhysX):**

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_publish_research_evidence.py' -v
PY=outputs/v120_baselines/conda_envs/uniskill/bin/python
$PY scripts/publish_research_evidence.py --source outputs/research_runs/b06_uniskill_stackcups_dev_v1 --dry-run
# Confirm file_names includes robot_rollout_preview.jpg before actual upload.
$PY scripts/publish_research_evidence.py --source outputs/research_runs/b06_uniskill_stackcups_dev_v1
```

The preview publisher reads only generated robot `episode_*/source_*/rollout.mp4`, never original TRAIN/EVAL videos; it uploads an <=850 KB labeled 4×6 JPEG contact sheet along with sanitized JSON/README. If `--dry-run` shows only `evidence.json` and `README.md`, `robot_preview()` silently failed to decode a video; preserve the failure, check video files and ffmpeg before claiming the video was reviewed. An uploaded preview will be at a **new evidence directory** printed in `TRO1_EVIDENCE_PUBLISHED`, not necessarily the earlier B06-run direct evidence directory.

Current scientific decision unchanged: 0/4 recorded B06 TRAIN-pair DEV success; actual motion failure mode **undetermined until the preview JPEG is visibly inspected**; do not modify or rerun B06 on the basis of a guess.
