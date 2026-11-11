# 2026-11-11 — M1-B: start real matched-goal capability experiment, not another audit loop

## Decision and observed assets

M0 is **CLOSED**, fully archived in commit `d5de864`: original
RoboReel video-conditioned pi0.5 Uniform ND PushCube 46/60, with actual
per-seed 14/20,13/20,19/20 and SHA256/per-trial archive.
Vid2Robot 3/60 previously archived and not rerun.

M1-A after corrected real server execution:
`local_robot_action_demos=2`, genuine Panda PushCube TRAIN H5 plus two
cameras, saved by old v120 downloader at
`outputs/v120_baselines/robot_train`;
`local_source_push_train_videos=6`;
`local_source_pick_train_videos=0`.
**No validated same-final-goal source Pick+Place video, and no target
blocker-clearing Push trajectory yet.** These are scientific prerequisites.

## B1 — Run THE OFFICIAL ManiSkill PickCube motion planner on PhysX

Do not use the unsuccessful hand-written `ScriptedPickSource`. Official
ManiSkill provides `PandaArmMotionPlanningSolver` and
`solvePickCube` (OBB-grasp pose, motion planning, gripper close, goal
move) in `mani_skill.examples.motionplanning.panda.solutions`.
Original upstream documentation:
https://github.com/mani-skill/ManiSkill/tree/main/mani_skill/examples/motionplanning/panda

A bounded, project-local source example wrapper has been committed:
`scripts/m1b_run_official_pick_reference.py`. It launches the
**installed original ManiSkill** CLI, at most 3 episodes, CPU PhysX,
headless RGB video, `pd_joint_pos` (as upstream uses), no implicit
retries, max 480 seconds. The original H5, JSON and MP4 are retained
under `outputs/m1b_official_pick_reference`.
The wrapper checks source_type `motionplanning`, episode success in
JSON, corresponding real H5 actions, and records whether any
episode succeeded and whether video was captured.

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m unittest discover -s tests -p 'test_m1b_official_pick_reference.py' -q
    python -m scripts.m1b_run_official_pick_reference

Successful evidence marker:
`M1B_OFFICIAL_PICK_REFERENCE_VERIFIED` with
`successes >= 1`, `any_video=true`,
`usable_reference=true`.
The actual expert H5/video and JSON provide the reference
for a source controller. **THIS IS THE ORIGINAL PICKCUBE GOAL, NOT
THE SAME-GOAL CROSS-BEHAVIOR TASK.** Do not report B1 alone as
successful M1 transfer.

If all 3 attempts fail, or MPLib faults, record the solver log;
**do not tweak five Python grasp constants or rerun repeatedly**.
Fallback once to the official recorded ~29.3MB PickCube motionplanning
expert trajectory (haosulab/ManiSkill_Demonstrations), with provenance.
That recorded task is also NOT automatically a same-final-goal pair.
ManiSkill official solution may not be the same version as the pinned
RoboReel code; preserve this implementation difference.

## B2 — Adapt the VERIFIED expert to the single SAME FINAL GOAL (physical)

After B1, build the additional PhysX environment on the already
working RoboReel `PushCubeEnvV2` environment, **not** by replacing
the 60-run M0 benchmark. Key semantic changes:

- Task cube center at `C_0`; goal center `G` has identical
  XY target and grounded final cube Z for source and target.
- Source robot can PICK the task cube, move it above a fixed obstacle,
  place at `G` and RELEASE. Adapt the same official
  motion-planning OBB-grasp/pick logic to this source env.
- Target robot is push-only, with gripper permanently open, and
  must achieve `G`; no picking/lifting allowed.
- Controlled static barrier has one narrow portal. A movable
  blocker occupies the only push-accessible portal. A direct push
  of the target is obstructed.
- To check causal necessity, target must physically move blocker
  out of the portal (with real contact), then physically move
  the task cube to `G`. Record both actual object trajectories
  and success, not planned XY waypoints alone.
- A paired seed is valid only when the source and target have
  **identical target-cube start, goal and obstacle scene**; the
  source must finish grounded and released; target must not grasp.
- Scene logic must document fixed/movable actor poses, sizes,
  friction, support/contact evidence, phase stamps, collisions,
  and goal predicate. Single pilot seed (e.g. 120) is DEV ONLY.

**Stop condition**: If the actual Push-only clear-then-push plan
cannot be physically executed, M1-B physics feasibility is
**BLOCKED**. Do not label hypothetical state-space plans as ground
truth; change the scene topology once using the measured contact
reason, not arbitrary dozens of seeds.

## B3 — Baselines FIRST, before M2 learned method

In the same scene and task goal, with identical observation budget:

1. **TAMP**: capable, planner-based oracle that selects
   `push blocker` -> `push target`; public RGB perception
   version separately from privileged-state sanity upper bound.
   If TAMP is fully successful, new editor novelty is at risk.
2. **Goal-only BC/DP**: genuine target TRAIN robot action data
   under matched M1 task, not only 2 unrelated RoboReel H5s.
3. **Raw source object flow + same low-level controller**.
4. **Geometry-only A* / waypoint and direct video-conditioned
   policy** with controlled inputs.
5. Verify actual Im2Flow2Act, EC-Flow and Cross-Behavior Object
   Flow implementation/port status, do not claim official score
   from unavailable code.

Counterfactual controls: open/blocked passage (insertion needed vs
not), grasp-enabled/push-only target (behavior reversal), optional
source detour, physically unreachable target (abstention).
The source video must convey some meaningful task information beyond
already completely specified target scene+goal; otherwise video
conditioning is unnecessary and the cross-behavior claim must be
reframed. Equal-information TAMP and goal-only tests decide this.

## Current formal status

- M0: PASS, GitHub results immutable.
- M1-A: PASS for data-presence; 2 real robot TRAIN demos.
- M1-B1 official reference: **CODE COMMITTED, PHYSICAL NOT_RUN**.
- M1-B2 same-goal scene and real blocker insertion: **NOT_RUN**.
- M1-B3 same-goal baseline physical comparisons: **NOT_RUN**.
- M2 method: NOT_STARTED.

No T-RO likelihood or novelty gain claimed yet. The full source
artifact is an executable and verifiable B1 reference, not a paper
algorithm. Need actual B2/B3 before modeling new effects.

## 2026-11-11 — v1.6.5 REPAIR: Real official PickCube run must be REUSED

Operator executed the wrapper's unittest and original three-episode
ManiSkill PickCube motion planner. Two independent wrapper bugs were
observed:
1. Old fixture mocked a completed reference without the
   `usable_successful_pick_reference` reporting field, causing
   `KeyError` in unittest.
2. The original solver subprocess reached `validate_record`
   (not a solver process return-code failure), but our validator
   assumed top-level `meta.env_id` and nested
   `episodes[i].info.success`. Upstream `RecordEpisode`
   actually publishes `meta.env_info.env_id` and
   `episodes[i].success`; its source is
   `mani_skill/utils/wrappers/record.py`. The schema bug generated
   `ValueError: Not the original ManiSkill PickCube-v1 demo metadata`.

The solver's actual success outcome has NOT yet been inspected.
**Do not re-execute the official planner.** H5/JSON/MP4 generated by
this first real run are reused.

Code repair:
- Verify environment from `env_info.env_id` and success at the
  official episode root `success`, optionally cross-check H5
  per-step terminal success.
- Assert H5 actions match JSON `elapsed_steps`.
- Allow 1–3 nonempty saved episodes from three attempts, since
  `RecordEpisode` may prune empty episodes. Do NOT invent missing
  successes or claim three saved.
- Add `--verify-only` to strictly prevent subprocess execution
  while inspecting existing `m1b_official_pick.h5` and `.json`.
  Verify before writing an immutable compact JSON report.
- Repair the fake unittest fixture and test no-rerun reuse.
- `usable_reference` requires observed source episode success and
  at least one actual RGB MP4. Video/episode identity alignment remains
  unproven unless the recording file mapping is also checked.
- Existing H5 or JSON are never deleted or overwritten.

Execute ONLY:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    python -m unittest discover -s tests -p 'test_m1b_official_pick_reference.py' -q
    python -m scripts.m1b_run_official_pick_reference --verify-only

Expected marker `M1B_OFFICIAL_PICK_EXISTING_VERIFIED`; inspect the
`successes` and `usable_reference` values, NOT merely process exit.
If source success is 0, report that scientific source-asset failure and
inspect `outputs/m1b_official_pick_reference/official_solver.log`
for the planner exception. Do not automatically iterate thresholds.

M1-B2 matched-goal Pick+Place and Push-only blocker insertion, and
M1-B3 fair baseline comparison, remain NOT_RUN until the source
reference is physically validated. This change is an evidence parser
fix only, not a learned transfer method.
