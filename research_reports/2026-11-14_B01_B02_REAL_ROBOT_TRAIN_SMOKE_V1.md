# TRO1 B01/B02 · Real RoboReel robot-demo training smoke V1

Date: 2026-11-14
Status: `SMALL_DEV_DATA_ONLY`; two authentic robot demonstrations; 100 optimizer steps per model; not formal baseline evidence; no PhysX evaluation in this run.

## Data verified by operator

Root: `outputs/v120_baselines/robot_train/robot_trajectories/no_distraction/push_cube`

| Demo | H5 actions | Panda articulation rows | Base frames | Hand frames | Resolution |
|---|---:|---:|---:|---:|---|
| 0000 | 138×8 | 139×31 | 138 | 138 | 512×512 RGB |
| 0001 | 136×8 | 137×31 | 136 | 136 | 512×512 RGB |

All H5 actions finite. Video frame counts exactly match action rows. Panda state has one extra row; preprocessing uses first N states to align with N actions. Exact temporal semantics of this leading/trailing state alignment remains to be checked against upstream collection code.

Preprocessing succeeded with label `SMALL_DEV_DATA_ONLY`. Split is demonstration-level (one demo train, one demo validation); no frame-level split leakage. This is too small to support claims about generalization.

## User-reported 100-step runs

### B01 Transformer BC reimplementation

- output: `outputs/research_runs/b01_smoke_v120_100step`
- checkpoint: `best.pt` (2.5 MB)
- train loss: 1.14377 at step 1 → 0.02730 at step 100
- reported validation loss: 2.20668 at step 1 → 0.0076459 at step 100

### B02 compact conditional DDPM reimplementation

- output: `outputs/research_runs/b02_smoke_v120_100step`
- checkpoint: `best.pt` (2.7 MB)
- train loss: 0.91939 at step 1 → 0.92594 at step 100
- lowest reported validation loss: 0.93823 at step 90
- DDPM objective is epsilon/noise-prediction MSE; it is not numerically comparable to B01 action MSE. A value near 1 is on the scale of unit-variance Gaussian noise prediction; this short run alone does not establish failure.

## Critical validation-code correction

Review after these runs found the first validator evaluated only the first 16 windows of the held-out demo. Thus the validation curves above are historical smoke observations, **not reliable full-demo checkpoint-selection metrics**. Existing local checkpoints are preserved, but do not use them for physical comparison or report their best validation loss as a full validation result.

Latest branch commits fix this:
- `fd29b3b`: evaluate every validation window; DDPM validation uses fixed seeded noise and timestep draws.
- `049c766`: integration test for full-window coverage and deterministic DDPM validation.
- `4e3ea2b`: runbook updated with correction.

Next action: pull branch, run full tests, repeat 100-step smoke to fresh output directories ending `_fullval_v2`. Only if the corrected run and checkpoint inference look sound should at most two non-heldout DEV PhysX rollouts per model be considered.

## Evidence boundary

- Models are `REIMPL_ON_ROBOREEL`, not official author ACT/DP implementations or checkpoints.
- Real 4090 training did occur for these two small-data 100-step jobs, as shown by user-provided logs.
- B01/B02 real ManiSkill PhysX evaluation: NOT RUN.
- Formal train dataset threshold (>=64 authentic robot TRAIN demos): NOT MET.
- Official 20×3 held-out evaluation: NOT RUN.
- No claim of baseline completion or comparative performance.
