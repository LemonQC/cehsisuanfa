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

## 2026-11-10 follow-up: confirmed *wrong network* DNS answers

On the user's server the system resolver returned:
- huggingface.co A = 108.160.170.45: ARIN allocates 108.160.160.0/20 to Dropbox.
- huggingface.co AAAA = 2a03:2880:f10e:83:face:b00c:0:25de: Meta/Facebook AS32934.

Public Google DNS observation for the same domain returned Amazon CloudFront
3.171.171.104 / 3.171.171.6 / 3.171.171.128 / 3.171.171.65 at the time
of investigation. These are **observations, NOT permanent DNS pins**.
Do not overwrite /etc/hosts with fixed CDN addresses.

Both direct Python TLS and curl failed before handshake: a connect timeout
is therefore expected; updating pip/certifi is not the right next step.
Likely causes include poisoned/wrong system or upstream DNS, a locally
installed hosts rewrite, a local DNS forwarding service, or corporate
routing. No proxy env variables appeared in the diagnostics.

Run read-only on server:

    grep -nE '(^|[[:space:]])huggingface[.]co([[:space:]]|$)' /etc/hosts || true
    cat /etc/resolv.conf
    (resolvectl status || systemd-resolve --status || true) | head -80
    getent ahostsv4 huggingface.co
    if command -v dig >/dev/null; then dig +time=3 +tries=1 @1.1.1.1 huggingface.co A; fi

A query to 1.1.1.1 may be blocked by a managed firewall; if so, failure
does not prove normal DNS and is not a reason to disable TLS. Use an
approved, trustworthy resolver/VPN/proxy configured by the network
administrator or an authorized local artifact transfer with integrity
checks. Verify renewed DNS and HTTPS before re-running downloads:

    python scripts/diagnose_hf_tls.py
    curl -4 -I --connect-timeout 5 --max-time 15 https://huggingface.co
    python -c "from huggingface_hub import HfApi; print(HfApi().model_info('LogosRoboticsLab/roboreel_models').id)"

If an independent trusted resolver provides a current AWS CDN IPv4
address, a *temporary diagnostic* with `curl --resolve` can establish
whether only DNS resolution is broken; keep HTTPS hostname and certificate
verification enabled, and never treat that address as a permanent hosts
entry. If even a trusted IP cannot be reached, fix outbound networking
rather than endlessly swapping DNS addresses.


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
