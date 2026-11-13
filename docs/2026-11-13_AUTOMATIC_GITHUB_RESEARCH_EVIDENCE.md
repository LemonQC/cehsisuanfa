# TRO1 Automatic Research Evidence Publishing (2026-11-13)

## Permanent location

- GitHub repo: `LemonQC/cehsisuanfa`.
- **Dedicated report branch**: `tro1-experiment-evidence` (created 2026-11-13).
- Report root: [`evidence/runs`](https://github.com/LemonQC/cehsisuanfa/tree/tro1-experiment-evidence/evidence/runs).
- Experiment/code branch remains **`main`**, which the publisher never writes to.
- Each result goes to a unique content-addressed immutable directory, e.g. `evidence/runs/drawer_hf_v1_<sha16>/`. Use commit/permalink for audit. Re-publishing the identical result does not create duplicate files, and changed local results create a new folder rather than overwriting older evidence.

## What is uploaded

`evidence.json`: sanitized numeric metrics from existing `results.json` (paired trial success, full mean 8D action L2, trial steps, error **type only**, source result SHA256), or an immutable partial result if interrupted. `README.md`: readable summary, success/episode table and disclaimer about DEV vs formal. Optional `robot_rollout_preview.jpg`: small contact sheet constructed **only from produced robot rollout MP4 files**, NOT the source human videos. Pipeline error stage and exit code (if wrapper ran) are safe fields. A `TRO1_EVIDENCE_PUBLISHED` line carries a clickable GitHub URL when successful.

**Explicit exclusions**: never upload raw `rollout.mp4`, any gated human demos, training inputs, PartNet or ShapeNet archives/meshes, SigLIP features, `.npy` action recordings, weights, checkpoints, auth tokens, Hugging Face caches, full logs, raw manifests, or arbitrary user files. No raw exception text, full local paths, arbitrary environment variables or credentials. Media previews can be disabled with `--no-preview`.

Local scratch reports stay on the project's 4 TB mount in `outputs/experiment_publisher/` (already gitignored); no large files go to system `/tmp` from the publisher.

## For the previously started or completed drawer_hf_v1 experiment: upload WITHOUT rerunning

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_publish_research_evidence.py' -v

# Works after success or failure; reads existing outputs only.
python -u -m scripts.publish_research_evidence \
  --source outputs/roboreel_video_dependency_dev/drawer_hf_v1
```

This requires `gh auth login` / previously working GitHub CLI credentials with **push permission**. Run `gh auth status` locally to inspect, do not paste tokens. If the branch is protected/unavailable, the publisher **fails clearly** and retains the local sanitized snapshot; it never falls back to pushing `main`.

## For future new drawer experiments: one command runs and publishes even on error

```bash
bash scripts/run_drawer_with_evidence.sh
```

The wrapper carries out PartNet asset inspection → scene-only smoke → authentic TRAIN data and model preflight → the existing 2×2 paired rollout. A shell EXIT trap writes a tiny pipeline stage status and invokes the sanitized publisher whether the experiment **succeeds or fails**. A completed prior `drawer_hf_v1` (manifest, results, episode directories or partial results) causes a **hard refusal** before any re-run; use the standalone publisher instead. It never re-downloads PartNet/ShapeNet, human videos or SigLIP; never trains; never re-runs immutable old evidence. The scene and experiment still run on the user's RTX 4090 only, never on ChatGPT's remote service.

## Offline validation

- `tests/test_publish_research_evidence.py` includes no-network cases for secret/path redaction, content-addressed identity, partial runs, blocked datasets, branch enforcement and no-overwrite semantics.
- The uploader can be checked without GitHub writes by:

```bash
python -m scripts.publish_research_evidence \
  --source outputs/roboreel_video_dependency_dev/drawer_hf_v1 --dry-run
```

`--dry-run` writes only `outputs/experiment_publisher/snapshots/` locally. The full remote publish uses GitHub CLI's installed authorization via its **local** credential store; passwords/tokens never appear in script arguments or JSON. GitHub has its normal repository access controls; a report branch in a **public** repo is publicly readable, so only approved redacted metrics are published.

## User/assistant workflow from now on

1. User runs future research experiments through an auto-publish wrapper or invokes the standalone publisher after an existing run; **no copy/paste of results needed**.
2. User need only say `结果已经自动推送` (or include the GitHub evidence path). Assistant reads `tro1-experiment-evidence/evidence/runs/...` using GitHub connector, including numeric results and robot visual preview, before decisions.
3. Assistant continues to submit research decisions and code to `main` and compares only real executed evidence; any missing outcome stays `NOT_RUN`. No baseline re-run merely for reporting.

**Security**: do not post gated source clips or credentials to the public branch, even if the project itself is private today. Publisher uses a conservative structured allowlist; do not expand it to arbitrary file upload.
