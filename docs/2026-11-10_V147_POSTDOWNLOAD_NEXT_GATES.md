# 2026-11-10 — v1.4.7 π0.5 Uniform post-download gate and next science checkpoint

Status: **the operator reports that checkpoint downloads finished**, but current
GitHub cannot inspect the RTX4090 local files. Official π0.5 physics remains
NOT_RUN. v1.3 official Vid2Robot (3/60) is archived and must not run again.

## Step 0: local file and existing protocol gate

On Ubuntu, in clean main after a fast-forward pull:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    conda activate agcofl4090
    python -m pytest -q tests/test_v141_pi05_assets.py tests/test_v140_official_pi05.py
    find outputs/v120_baselines/released_checkpoints/nd/vid_cond_pi05_uniform \
      -type f | wc -l
    du -sh outputs/v120_baselines/released_checkpoints/nd/vid_cond_pi05_uniform
    python -m agcofl.v141_pi05_assets --variant uniform

Full official selection includes 24 method-local files + a repository README
(25 total). Confirm Orbax params AND train_config.json are present. Existing
preflight will still show siglip_feature_0080..0099 while their local
bundles have not been computed. **Do not re-download the checkpoint.**
Local file presence alone does not prove checkpoint tensor restore works.
If any other gate fails, inspect the report instead of modifying model metadata.

## Step 1: separate original authors' OpenPI/JAX environment

The current agcofl4090 environment (which executed the Python/HF downloader)
is NOT established as the original π0.5 inference environment. A separate
conda env such as pi05 is expected; use pinned original authors'
third_party/roboreel_models/ENVIRONMENTS.md section 1.

    conda env list
    conda activate pi05
    python -c 'import jax; print(jax.__version__, jax.devices())'

For the original source tree, set:

    export PYTHONPATH="$PWD/third_party/roboreel_models:$PWD/third_party/roboreel_models/src:$PWD/third_party/roboreel_models/src/external_models:$PWD/third_party/roboreel_models/src/external_models/lerobot/src:$PWD/third_party/roboreel_models/roboreel_env/src:${PYTHONPATH:-}"

Do not invoke paid LLM/VLM annotation APIs. The official SigLIP tokenizer
may require a separate (not the policy checkpoint) released weight download;
if so, report its exact log. Never substitute random weights/heuristic cache.

## Step 2: one real EVAL video feature smoke (NEW)

    bash scripts/run_v141_next_gate.sh smoke-one-feature

This runs the original author SigLIP featurizer on *only one* missing
EVAL video, through a new explicit --max-feature-demos 1 limiter. It writes
a bundled video_demonstration_features/without_distraction/push_cube/0080/front.npy
(or the first missing ID). It is **evaluation-time input processing, not
training on the evaluation videos**. Partial asset gate still prints
ready=false, which is expected.

Then finish the remaining EVAL videos:

    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready

The final command exits nonzero unless all original checkpoint/protocol
checks and *all 20* authentic SigLIP bundles pass the gate. No weights
are retrained or evaluation data used for checkpoint selection.

## Step 3: exact official ND 20x3 physics — ONLY after ready=true

    bash scripts/run_v141_next_gate.sh evaluate-uniform

This is a complete official 3-seed evaluation (0,100000,200000) with 20
held-out video IDs 0080..0099 in each seed bank, **60 genuine PhysX
rollouts**, max_steps=800, original action_horizon=20, unchanged official
success flags. Per-run logs:

    outputs/v140_pi05_uniform/seed_offset_0/run.log
    outputs/v140_pi05_uniform/seed_offset_100000/run.log
    outputs/v140_pi05_uniform/seed_offset_200000/run.log

Machine-checkable official summary on successful completion:

    outputs/v140_pi05_uniform/official_aggregated_summary.json
    outputs/v140_pi05_uniform/status.json

If a run fails, inspect its run.log and keep partial JSON; never mark a
started or preflight-only job as a successful baseline. Existing summary
files verified by the script are not rerun. After execution, publish only
verified small result JSON and method details to research_reports, never
checkpoint blobs or source videos to GitHub.

## Step 4: scientific main track (independent of official π0.5 evaluation)

    conda activate agcofl4090
    bash scripts/run_v141_next_gate.sh custom-physics-dev

A scripted privileged-state source Pick/Lift over fixed obstacle and
target Push-only physics pilot. FIRST verify *real* valid paired source
grasp, overflight, release, goal success and target support/contact.
The pilot is NOT a trained policy. If physics fails, fix task validity
before running learning baselines.

Next build genuinely learned effect editing + closed-loop policy and
compare against Goal-only, Raw Flow, Waypoint Oracle and capable
task-and-motion planning under matched observations and TRAIN budgets.
Include target-only necessary blocker clearing, not just geometric
detours. The paper story is in
docs/2026-11-10_TRO_PAPER_STORY_CAPABILITY_EFFECT_ADAPTATION_PROPOSAL.md.
