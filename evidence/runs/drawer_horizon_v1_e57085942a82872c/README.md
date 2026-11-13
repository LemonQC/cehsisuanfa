# TRO1 automated, sanitized experiment evidence

- Task: `toys_in_drawer`
- Status: `READ_ONLY_DRAWER_HORIZON_AUDIT` (may be DEV; not paper-ready)
- Source in project: `outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1`
- Reported PhysX rollouts: `None`
- Paired episodes: `None`
- Any pair differs in success: `None`
- These are *copied numeric findings*, not new runs; raw video, model and gated datasets are excluded.

## Per-trial results

| Episode | TRAIN demo ID | Steps | Success | Runtime error type |
|---:|---:|---:|---|---|
| — | — | — | — | — |

## Matched pair action sensitivity

| Episode | Success outcomes | Full mean 8D action L2 |
|---:|---|---:|
| — | — | — |

## Artifact verification

- `evidence.json` contains source SHA256 hashes and the detailed sanitized metrics.
- All uploads are snapshots: existing changed evidence files will never be overwritten.
- This branch is dedicated to measurements; the repository `main` branch is unaffected.

## Read-only 400-step vs 800-step success audit

- Recorded success within first 400 steps: **0/4**
- Recorded success within first 800 steps: **2/4**
- Prefix audit of frozen DEV trajectories; NOT official heldout 400-step evaluation.

| Episode | TRAIN source | Steps | First success step | By 400 | By 800 |
|---:|---:|---:|---:|---|---|
| 0 | 0 | 800 | — | False | False |
| 0 | 1 | 800 | — | False | False |
| 1 | 0 | 634 | 634 | False | True |
| 1 | 1 | 650 | 650 | False | True |
