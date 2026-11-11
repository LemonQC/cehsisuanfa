# 2026-11-11 — M0 passes. M1 PRE-METHOD GATE (frozen research task and competing baselines)

## 0. M0 outcome, now operator verified

**Original RoboReel video-conditioned π0.5 Uniform, clean/clean PushCube: 46/60 = 76.67%.**

Evidence: operator pasted `PI05_SELECTIVE_ASSET_GATE ready=true` for all
20 EVAL front videos/embeddings, CUDA `gpu:0` runtime checks, and
`ACTUAL_PI05_PHYSICS_VERIFIED {"variant":"uniform","k":46,"n":60}`.
The program prints this marker only after validating three distinct
seed offsets (0, 100000, 200000) × demo IDs 0080..0099, checking
actual original-author per-trial summaries and storing the aggregate.
Model code pinned at `79a0fb0099e73c5560114753a8706276f93aefc2`.

Existing Vid2Robot official clean/clean PushCube is **3/60 = 5.00%**;
do not rerun it.

**Audit boundary**: the user-server run was verified by the existing
evaluation program; the compact 60-row raw archive is not yet pushed
to this GitHub repo. `scripts/archive_pi05_uniform_m0.py` was added
to verify all original local per-seed summaries and export
`research_reports/2026-11-11_m0_pi05_uniform/` (per-trial CSV,
validation JSON, SHA256 provenance). Its execution is still pending
on the operator's server. Third-party mirrored OpenPI base SigLIP
checkpoint has proven *functional restoration*; exact binary identity
to the public GCS origin has not been independently confirmed.
Preserve the original files, no checkpoint/video redownload.

**Conclusion:** the M0 official pipeline is operational. Neither the
large gain of π0.5 over Vid2Robot nor a 76.67% success rate proves
any scientific novelty for our capability adaptation method.

## 1. Paper question for M1

Under **the identical final object goal**, and given a human/source
demonstration that uses a physically unavailable interaction mode,
does target-aware **causal effect editing** provide a measurable
advantage over flow imitation, goal-only closed-loop policies,
and an **equally informed** capability-aware task-and-motion planner?

Not merely whether an arm can grasp a cube, trace a 2D detour, or
follow a human source flow.

## 2. Frozen M1 experimental axes

**Dataset split and fairness:**
- Build CUSTOM-ADDITIONAL scenes on the pinned ManiSkill/PhysX
  benchmark infrastructure; do NOT relabel official PickCube
  and PushCube as same-goal tasks (they have different success
  conditions).
- For each paired example, enforce identical task goal, initial
  target physical scene, object identities, target sensing and
  fixed target controller/policy action interface.
- Source video is **actual** source interaction footage from a
  verified expert, teleoperation recording, or validated existing
  motion assets. It must achieve the exact same *final object-state
  goal* as the target, even if it uses Pick rather than Push.
  Synthetic waypoints alone are not human video.
- TRAIN/validation/test scenes use disjoint object/layout seeds.
  Never use official EVAL 0080–0099 to train, tune, or select.
- In the RGB leaderboard, methods only receive observed RGB(D)
  and published camera/capability metadata (no test-time simulator
  object-state oracle). Privileged-state planning is a **separate
  upper bound**, not a direct score in the RGB table.

**Four frozen task regimes:**
- **R (Replace):** source Pick+lift over barrier, target Push-only
  route around obstacle while reaching same target cube goal.
  This is a weak sanity task (standard planning may solve it).
- **I (Insert):** source lifts object over obstacle, whereas target
  Push-only must **first move a movable blocker out of a passage**
  before pushing the task object to the *same* final goal.
  Inserting this previously unseen target-only causal step is the
  primary scientific test; use real physics, not plotted waypoints.
- **D (Delete):** source contains irrelevant handling/detour;
  target should ignore it while preserving task success.
- **U (Unreachable):** no allowed target action sequence reaches
  the goal; a calibrated reject/abstain decision must be tested,
  not counted as random rollout success.

**Causal counterfactuals** (pre-register before training):
- Same human video, final goal and scene; switch only target
  capabilities (grasp enabled versus push-only). A valid method
  must choose physically distinct effects where necessary.
- Same goal/capability, vary whether the blocker occupies the
  only push passage; insertion should appear/disappear accordingly.
- Same video/task and incidental detour changed; essential
  effects should stay unchanged.
- Separate fully observable targets from cases where video conveys
  genuinely needed task information (unmentioned constraints,
  identity, procedural order). For fully specified, fully
  observable goal/state tasks, **do not promise source video will
  beat goal-only**. The source-information advantage must be
  evaluated, not assumed.

## 3. Baselines that MUST be evaluated before M2 model design

| Order | Control | What it tests | State |
|---|---|---|---|
| 1 | Goal-only BC/DP, no human video | What target-only imitation already solves given identical robot TRAIN labels | NOT_RUN on same-goal M1 |
| 2 | Raw source-flow + same downstream controller | Does direct flow fail because the contact/support mode is unavailable? | NOT_RUN |
| 3 | Geometry-aware waypoint / replanning | Is a path change sufficient without new causal events? | NOT_RUN |
| 4 | Capability-aware TAMP (perception + effects + allowed actions) | Does ordinary planning already solve target-only insert/reject? | NOT_RUN |
| 5 | Source video + strong direct policy (original π0.5 if port is validated, otherwise transparently reimplemented video-conditioned BC/DP) | Can an existing model learn appropriate behavior without an explicit editor? | NOT_RUN on M1 |
| 6 | Im2Flow2Act / EC-Flow / Cross-Behavior Object Flow, faithful author checkpoint or *documented* port | Strongest related flow transfer comparison | CODE/PORT STATUS MUST BE VERIFIED BEFORE CLAIMING RESULTS |

**Do not claim baseline completion because the script is present.**
Existing official RoboReel π0.5 Uniform 46/60 and Vid2Robot 3/60
are **M0 benchmark evidence only**, NOT M1 control performances.
No need to rerun M0 as an M1 baseline when goal/action/task are not
the same. Strong TAMP must be compared using matched public scene
perception/geometry/capability, with a privileged-state oracle reported
separately. If an information-matched TAMP solves all blocker-insertion
cases, the editor's novelty claim must be narrowed or rejected.

## 4. M1 work order and acceptance criteria

1. **Now:** Archive the genuine M0 60 rows. Small CSV/JSON/MD only.
2. **Next:** Inventory genuine target TRAIN action trajectories and
   validated source demonstration assets (RoboReel, ManiSkill, other
   author-released datasets). Decide a physically valid paired task
   construction and record deviations. This is a dataset selection
   and reproduction step, NOT new algorithm invention.
3. **Then:** Run a small *fixed* 4-regime physical smoke on same-goal
   target scenes, using validated source trajectories + target robot
   actions, and implement controls 1–4 at equal observations.
   Preliminary smoke only; success requires actually observed object
   goal, intermediate blocker displacement and physical feasibility.
4. **Gate A PASS:** source and target produce real same-goal successful
   trajectories for R/I; U case physically impossible under allowed
   actions; capability/blocker counterfactuals reverse required effects.
5. **Gate B PASS:** video carries demonstrably relevant task context
   beyond the explicit goal, or the narrower claim is reformulated
   as capability-aware planning with demonstrated generalization.
6. **Gate C PASS:** stronger matching-video/state TAMP, goal-only and
   raw-flow controls leave a *measured* failure mode for effect editing
   to address; no paper-ready claim before this gate.
7. Only after A/B/C are supported, design and train the M2 method,
   then evaluate on disjoint held-out scenes with uncertainty.

**No more blind hand-written Pick-teacher tuning cycles.** The previous
seed120 scripted teacher failed actual grasp/lift; it is neither a
trained method nor a prerequisite if validated source assets are
available. The research object is causal effect adaptation.

## 5. Next server commands

**2026-11-11 runtime correction:** the user confirmed `python -m pytest` is unavailable in `pi05`, and direct `python scripts/archive_pi05_uniform_m0.py` failed because Python searched only the `scripts/` package path. Neither error touches the existing M0 46/60 results. This archiver has no pytest runtime dependency; use module invocation. To run its optional pytest tests, use an environment where pytest is installed, without changing the official pi05 env.


    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    # pytest is NOT installed in the pinned pi05 runtime and is NOT
    # required for this standard-library-only archive operation.
    python -m scripts.archive_pi05_uniform_m0
    # Direct python scripts/archive_pi05_uniform_m0.py also works after
    # the package import-path fix, but -m is preferred.
    # Verify small report directory; don't add models or outputs.
    git add research_reports/2026-11-11_m0_pi05_uniform
    git commit -m "results: archive original π0.5 Uniform 46-of-60 trials"
    git push origin main

If the server's Git main has diverged, stop and reconcile normally;
do not force push. After this one-time M0 provenance save, focus on
M1 source/target TRAIN dataset selection, baseline coding, and actual
matched-goal physical validation — not more original-checkpoint work.

## Paper standard

M0 demonstrates a successfully replicated **published** model.
M1 must establish a *novel unmet problem* under fair information.
M2 must provide a learned method and quantitative advantage on
the same task protocol. T-RO acceptability is not guaranteed by
any isolated score.
