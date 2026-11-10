# 2026-11-10 v1.4.8 — Pi0.5 OpenPI/JAX environment gate

## Real evidence from current server log

The operator's CPU tests passed 16/16. The official pi0.5 Uniform asset
gate now fails ONLY for SigLIP features for EVAL 0080–0099; the checkpoint
and train_config path checks pass. This does not prove Orbax tensor restore
or checksum verification yet.

conda env list under /home/fu/SoftWare/Anaconda has NO named pi05 env, so
conda activate pi05 failed correctly. It DOES list a prefix at
/home/ubuntu/miniforge3/envs/rokae_pi05; this could be an old/stale
environment from a different system user. Check existence before using it.
DO NOT alter the existing agcofl4090 or another project's rokae_pi05.

## If existing prefix is genuinely present and authorized to use

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    ls -ld /home/ubuntu/miniforge3/envs/rokae_pi05/bin/python
    /home/ubuntu/miniforge3/envs/rokae_pi05/bin/python -V
    conda activate /home/ubuntu/miniforge3/envs/rokae_pi05
    bash scripts/run_v141_next_gate.sh check-pi05-env

This is read-only. If the path doesn't exist, skip it.

## Otherwise create a separate original pi05 environment

    conda create -n pi05 python=3.11 -y
    conda activate pi05
    bash scripts/setup_pi05_author_env.sh
    bash scripts/run_v141_next_gate.sh check-pi05-env

The opt-in setup script implements section 1 of the PINNED official
third_party/roboreel_models/ENVIRONMENTS.md and refuses to install unless
CONDA_DEFAULT_ENV=pi05 and Python=3.11. It does not need a paid LLM API.

## Once JAX CUDA and imports are confirmed

    bash scripts/run_v141_next_gate.sh smoke-one-feature
    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Smoke: original SigLIP applied to ONE held-out EVAL video for inference
conditioning, not policy training. Full prep fills only missing 20 EVAL
video feature bundles. Evaluate only after ready=true and author runtime
check; 3 seed banks x 20 physical rollouts, max_steps=800, original
action_horizon=20. Do not run Vid2Robot 3/60 a second time.

## Independent scientific task

    conda activate agcofl4090
    bash scripts/run_v141_next_gate.sh custom-physics-dev

This is same-goal Pick-source -> Push-only target SCRIPTED PhysX feasibility,
not yet a learned policy or evidence supporting the T-RO claim.

Status: user has run preflight and 16 unit tests; newly committed JAX
runtime check/installer/tests have NOT been executed on the user's RTX4090.
