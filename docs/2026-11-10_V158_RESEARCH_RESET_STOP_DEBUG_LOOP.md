# 2026-11-10 — Research reset: STOP engineering/debug loops; return to T-RO novelty test

## Decision
The current work has drifted from the T-RO scientific hypothesis to debugging a newly scripted source Pick controller and installing auxiliary baselines. **STOP this loop now.** No more iterative tuning of `ScriptedPickSource`, repeated seed120 runs, blind installation fixes for official π0.5, or expanding baselines until the novelty question is answered.

## What the evidence actually says
- Official RoboReel Vid2Robot v1.3: previously verified 3/60, not to rerun. This alone does **not** prove novel cross-behavior capability adaptation.
- Official π0.5 Uniform: fine-tuned checkpoint downloaded, but authentic base SigLIP cache absent and public EVAL feature bundles absent; 0 official physical trials. Mark **BLOCKED**, do not misreport success.
- Custom Pick→Push: one real seed120 source rollout failed (281 steps, peak lift 0.000088m, no overflight); targets not executed. Code can be useful infrastructure but is **not** evidence for algorithm novelty. Do not keep treating a source-scripted teacher as a research deliverable.
- Neither successful scripted trajectories nor their failure can establish an advantage over equal-information goal-only policies or task-and-motion planning.

## Claim to falsify first
**Capability-aware causal effect adaptation**: when the target embodiment cannot physically execute source object-motion/contact effects, the robot identifies the goal-relevant constraints in the source video and learns to *replace, delete, insert,* or *reject* effects while preserving the same object goal. In particular, some target tasks require *new target-only prerequisite effects* absent from the source, not just 2D obstacle detours.

## THREE scientific gates BEFORE implementing a new model

### Gate A — problem necessity
Specify matched-goal, matched-observation/source-video examples; source and target must have the SAME desired final object state. Include:
1. source lift-over-barrier vs target push-around-barrier (replace)
2. target needs to move a removable blocker first (insert a new causal prerequisite)
3. truly unreachable under target capability (reject)
Counterfactual intervention: keep source demonstration, goal, scene fixed and change *only* target capability; expected adaptation must change.

Do NOT implement a custom teacher from scratch to establish Gate A. Reuse existing validated simulation trajectories/controllers and short recorded counterfactual state/action examples; skip physical teacher debugging until problem semantics and baselines are checked.

### Gate B — source demonstration carries information
Compare video-conditioned model with goal-only baseline given the SAME target observations and goal. Clarify what information video conveys that is not already in the supplied goal/state. If video is irrelevant, the task is generic TAMP and does not support the intended cross-behavior paper story.

### Gate C — novelty beyond strong planning
Compare a raw object-flow follower, geometry/waypoint optimizer, a capability-aware TAMP planner using the SAME accessible state and action affordance information, and a goal-only policy. Privileged state oracle is a separately labeled upper bound. If a simple capability-aware planner already achieves comparable adaptation and behavior, a learned effect editor adds no demonstrated algorithmic value and the idea must be redesigned or STOPPED.

## Work sequence
1. Write **a 1–2 page scientific protocol**: exact input/output, where source information enters, what capability changes, matched goal/scene, causal effect edit, precise success conditions and 3 gates.
2. Audit code and existing measured results against the gate requirements; classify each baseline as actual / partial / blocked, not paper results.
3. Only after Gate A and B pass, prepare the **smallest** simulator demonstration needed to discriminate learned causal editing from strong equal-information TAMP and flow imitation. No large-scale model training or arbitrary debug until then.
4. If no source-specific advantage or no nontrivial target-only effect insertion can be demonstrated, **terminate/pivot** rather than spending weeks on environment engineering.

## Rules
- Never claim T-RO acceptance is guaranteed.
- Do not call scripted privileged-state teachers a new learned policy.
- Do not weaken physical validity or fair information budgets to manufacture an innovation gap.
- Document evidence separately from hypotheses; archive to this repo.
