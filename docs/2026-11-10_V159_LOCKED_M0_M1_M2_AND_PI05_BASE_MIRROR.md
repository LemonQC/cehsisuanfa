# 2026-11-10 — v1.5.9 SINGLE FROZEN EXECUTION TARGET: faithful baselines + falsifiable T-RO research

This document supersedes the contradictory advice to "stop downloading checkpoints" in
`docs/2026-11-10_V158_RESEARCH_RESET_STOP_DEBUG_LOOP.md`. It does NOT
rewrite or erase prior measured results. **Needed checkpoints must be
downloaded**. But every download, experiment and debugging step must
support a frozen acceptance criterion below.

## Research target — paper, not a scripted control demo

Working paper question:
**Can a robot learn task-relevant object effects from a source human
video and rewrite the effect/causal sequence for an embodiment that
cannot execute the source behavior, then execute the *same goal* with
a learned closed-loop robot policy?**

We must show at least target-only causal effects missing in the source
(e.g. first clear a movable blocker), not merely a 2D waypoint detour.
We must also prove source-video information helps relative to a strong
goal-only model and equal-information capability-aware task-and-motion
planning. No claim of novelty based only on flow extraction, frozen ViT,
path planning, hand-crafted teacher, or a successful source grasp.

## Workstream 1 — OFFICIAL benchmark baseline (complete it faithfully)

**M0 deliverable:** one real official RoboReel video-conditioned
π0.5 Uniform PushCube ND 20x3 physical success report, checkpoint config,
every held-out trial success, official JAX/OpenPI runner and exact
authors' SigLIP video embedding format. Vid2Robot 3/60 already measured;
do NOT rerun it.

Already present: released **fine-tuned ND Uniform ≈6.2 GB** checkpoint,
20 held-out EVAL front videos, JAX CUDA gpu:0. Missing: genuine
OpenPI pi05_base visual parameters and thus 20 EVAL SigLIP bundles.
No official π0.5 physics yet. The original author tokenizer needs
`pi05_base/params["PaliGemma"]["img"]`, exporting
`$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`.

**Critical new discovery:** The HF repository
`robotgeneralist/openpi_checkpoint_mirrors2`, revision
`1fda6b04e83d625eb1a55d68db32050e248cac2b` publicly lists
`pi05_base/params` (≈12.4 decimal GB), described by its maintainer
as a mirror of `gs://openpi-assets/checkpoints/pi05_base/params`.
The original GCS route failed at 0/11.6 GiB on this server. A public
HF mirror is a reasonable way to transfer necessary ORIGINAL-model
weights without a cloud account; however third-party provenance is
not independently proven equal to GCS. Verify metadata, file completeness,
Orbax restore and SigLIP outputs; record mirror revision. Keep original
author model code untouched.

**One frozen execution chain (NOT optional variants):**

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05

    # Metadata only; no download.
    bash scripts/run_v141_next_gate.sh plan-pi05-base-hf

    # Explicit model-weight download ~12.4GB, resumable within same dest;
    # small _METADATA probe before large transfer. Never fetch other models.
    bash scripts/run_v141_next_gate.sh prepare-pi05-base-hf

    # Same project-local cache path supplied to original author's code.
    bash scripts/run_v141_next_gate.sh check-siglip-cache
    bash scripts/run_v141_next_gate.sh smoke-one-feature
    bash scripts/run_v141_next_gate.sh prepare-features
    python -m agcofl.v141_pi05_assets --variant uniform --require-ready
    bash scripts/run_v141_next_gate.sh evaluate-uniform

The runner exports OPENPI_DATA_HOME under
`outputs/v120_baselines/openpi_cache` so the original
`maybe_download(gs://.../params)` short-circuits to the verified
local pinned HF mirror symlink (no GCS). Its original SigLipTokenizer
then exports its genuine source-derived visual encoder pickle locally;
NO silent replacement with fine-tuned Uniform `PaliGemma.img`.

If hf-mirror fails on the small blob probe, use a working approved
HTTPS proxy to `huggingface.co` by setting
`HF_ENDPOINT=https://huggingface.co`, or transfer the pinned
checkpoint from another machine. Do not repeat 11.6 GiB failed GCS
downloads or rerun the Python/CUDA dependency installer. Retain partial
HF files and resume exactly the same command after connectivity repair.
Do not move a partial Orbax cache to the location used by SigLIP.

M0 **PASS:** author SigLIP pickle genuinely restored, 20 exact EVAL
front.npy bundles read correctly, pi05 checkpoint restored,
60 physical official trials verified and archived. M0 remains BLOCKED
if any of these checks fail; this is a resource blocker, not a failed
scientific hypothesis.

## Workstream 2 — PRIMARY paper scientific necessity and scene protocol

**M1 deliverable:** one frozen CUSTOM-ADDITIONAL data/eval spec with
same task goal, same initial target scene and target robot observations,
source RGB human-style video, target capability variable and actual
robot action trajectories in TRAIN. Source Pick-lift and target Push
must have matching final object configuration; official RoboReel
PickCube (lift goal) and PushCube (XY goal) do not.

The primary discriminating task is the **necessary target-only effect
insertion** scenario:
source Pick over barrier -> target Push-only must FIRST move movable
blocker, THEN push marked object to common final goal. Include simple
replace, delete optional detour, and correctly reject unreachable.
Multiple held-out layouts/masses and counterfactual capability flips
are required before final claims.

Keep goal-only, raw object flow, geometric/waypoint, AND
capability-aware TAMP baselines. Equalize target obs/robot action
labels/training data; report privileged simulator-state oracle separately.
A source-only video vs goal-only comparison must explicitly define
what *new task information* a source demo transmits. If the goal and
fully observed target scene already determine all actions, source video
may not add scientific value: expose that, do not conceal it.

**No more from-scratch scripted source-controller threshold tweaking**.
The seed120 PhysX custom source failed (281 steps, 0.088mm lift, no
overflight, no target trials), and all source success criteria remain
strict. Use validated existing source/robot motion assets or a
validated controller / trajectory for a small paired physical scene;
if impossible, label M1 blocked and explicitly justify scene redesign.
Physical task validity is required, but is an enabler, not the claimed
novel algorithm.

## Workstream 3 — actual learned method, not an oracle script

**M2 deliverable:** source RGB effect graph -> capability-conditioned
effect editor (retain/delete/replace/insert/reject) -> trained closed-loop
target policy, plus 3 strong compared policies sharing exact target TRAIN
samples and robot observations. Labels for causal necessity and target
feasibility come from *TRAIN physically realized interventions*, not
test-time privileged ground truth or hand-coded policy labels.
Frozen local ViT allowed; no paid LLM/VLM APIs required.
Key ablations: no source video, no target capability, no causal insertion,
flow-only with same low-level executor, no receding-horizon correction.

**M2 scientific GO:** real target robot simulator trajectories achieve
same-goal target-only insertion and capability-dependent preference
reversal; learned editor improves over strong equal-information TAMP
and direct flow, with independent holdout scenes and error bars.
**Scientific STOP/PIVOT:** source gives no helpful information over
goal-only, or simple capability-aware planning matches method under
equal evidence and resources, or valid physical pairing cannot be built.
These are scientifically meaningful failures; routine dependency errors
are NOT a reason to abandon the paper direction.

## Research bookkeeping

Publish raw success k/N, seeds and split IDs, model provenance and
code differences. Keep offline technical tests and physics trial counts
distinct. Source checkpoint downloads do not equal completed baselines.
GPU runtime checks do not equal SigLIP tensor validation. Never report
an unexecuted method as having measured success.

This document is the single priority and acceptance reference for
subsequent research/engineering work unless the user explicitly changes
the scientific aim.
