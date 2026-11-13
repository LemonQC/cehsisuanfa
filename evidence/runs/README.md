# TRO1 experiment evidence (automatic, sanitized)

This branch is reserved for small, immutable results uploaded by the operator's research server, separate from experiment code on main.

- Reports: `evidence/runs/<experiment_name>_<sha256-prefix>/`
- Per report: `evidence.json` (numbers/source hashes), `README.md` and optionally a small `robot_rollout_preview.jpg` (ROBOT policy footage only).
- Never uploaded: HF tokens, human source videos, PartNet/ShapeNet models, policy weights, raw MP4s, NPY data, full logs, or credentials.
- Immutability: a changed report is never overwritten; it gets a new content-addressed folder.

Existing run: `python -m scripts.publish_research_evidence --source outputs/roboreel_video_dependency_dev/drawer_hf_v1`

New future drawer run: `bash scripts/run_drawer_with_evidence.sh`. It refuses to rerun already completed outputs.

Reports are execution evidence, not proof of scientific novelty. Missing and failed results remain explicit.

Instructions: https://github.com/LemonQC/cehsisuanfa/blob/main/docs/2026-11-13_AUTOMATIC_GITHUB_RESEARCH_EVIDENCE.md
