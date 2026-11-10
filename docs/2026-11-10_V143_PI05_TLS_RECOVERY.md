# 2026-11-10 — Official π0.5 Uniform download TLS failure and fail-closed recovery

## Evidence from RTX4090 operator log (NO physics)

The original authors' scripts/download_checkpoints.py calls
HfApi().model_info('LogosRoboticsLab/roboreel_models', files_metadata=True)
*before* any weights are downloaded. TLS fails with:

    SSLCertVerificationError: Hostname mismatch,
    certificate is not valid for 'huggingface.co'

The following evaluate-uniform preflight returned ready=false, because both
checkpoint params/train_config and 20 SigLIP front.npy EVAL features were
missing. The official evaluator did not execute any physics rollouts.
This failure is environmental HTTPS/certificate routing, NOT evidence that
π0.5 Uniform succeeds or fails on PushCube. Source v1.3 Vid2Robot 3/60 is
unchanged and must not be rerun.

## Code corrections (v1.4.3)

- scripts/run_v141_next_gate.sh evaluate-uniform now runs
  agcofl.v141_pi05_assets --variant uniform --require-ready first. An
  incomplete checkpoint/feature/protocol preflight exits 2 and prevents any
  official evaluator launch.
- scripts/run_v141_next_gate.sh prepare-checkpoint and prepare-features are
  separate, resumable stages. prepare-uniform runs the same stages in order.
  The original authors' selective ND Uniform downloader is unchanged.
- scripts/diagnose_hf_tls.py inspects direct TLS, env-aware requests, DNS and
  redacted proxy origins without printing credentials or downloading weights.
- tests/test_v141_pi05_assets.py adds a CPU-only negative gate test. This
  change was committed via GitHub; its tests are NOT CLAIMED EXECUTED on RTX4090.

## Recommended runbook

After verifying a clean server branch and pulling main:

    conda activate agcofl4090
    python scripts/diagnose_hf_tls.py

Review whether proxy hostname, DNS records or upstream HTTPS interception
are causing the wrong domain certificate. Also optionally run:

    grep -n -i 'huggingface.co' /etc/hosts || true
    getent ahostsv4 huggingface.co | head -5
    curl -Iv --max-time 15 https://huggingface.co 2>&1 | grep -E 'subject:|issuer:|SSL certificate|proxy|CONNECT|certificate'

If the requests path fails but direct TLS succeeds, review proxy env vars and
institution's network proxy configuration. If both paths fail, check DNS
and network routing / HTTPS interception before Python package changes.
Never use verify=False, PYTHONHTTPSVERIFY=0, altered hostname assertions or
a malicious CA certificate as an expedient fix.

Test TLS again before attempting any multi-GB downloads:

    python - <<'PY'
    from huggingface_hub import HfApi
    print(HfApi().model_info('LogosRoboticsLab/roboreel_models').id)
    PY

After HTTPS routing is corrected, prepare the original released parameters:

    bash scripts/run_v141_next_gate.sh prepare-checkpoint

Switch to a separate official JAX/OpenPI pi05 environment for the local SigLIP
encoder; agcofl4090 is a Vid2Robot/PyTorch environment, not validated for
official π0.5 inference. The authors' pinned ENVIRONMENTS.md defines the
required packages and paths:

    bash scripts/run_v141_next_gate.sh prepare-features
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Inspect ready=true from the gate before running a physical evaluation.
No keyframe API, no VLM calls, no fake training or new result claims.

If the required checkpoint is already downloaded and verified elsewhere,
use a local copy at:
    outputs/v120_baselines/released_checkpoints/nd/vid_cond_pi05_uniform/ckpt
with real params and train_config.json; do NOT fabricate empty files to
satisfy preflight. Reuse official SigLIP features only when they correspond
exactly to EVAL 0080–0099 and the original encoder.
