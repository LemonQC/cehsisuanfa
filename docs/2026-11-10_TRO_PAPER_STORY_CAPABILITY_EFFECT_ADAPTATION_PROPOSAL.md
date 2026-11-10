# 2026-11-10 — T-RO conditional paper storyline after baseline completion (PROPOSAL, NOT AN ACHIEVED RESULT)

> Status: prospective scientific framing only, conditioned on successful and faithfully documented baselines. No claims of trained novel policy, same-goal physical results, or guaranteed T-RO acceptability. This document records the **current conversation's** hypothesis and design alternatives; it does not import decisions from unrelated projects.

## Proposed one-sentence thesis

**Learn the task-relevant object effects from a demonstration rather than its precise flow, and adapt those effects into a feasible, dynamically realizable sequence under target capability and support/contact constraints, including deleting irrelevant source effects and introducing required target-only effects.**

The paper must not claim that robot independence of object flow is new: *Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation* (IEEE T-RO 2026, DOI 10.1109/TRO.2026.3701576) already addresses that. Im2Flow2Act (CoRL 2025) already uses flow-conditioned policies. EC-Flow (ICCV 2025) already considers embodiment kinematics and flow. Our contribution, **if demonstrated**, must be the inference and executable revision of object-state effects under *behavior-changing* constraints and causal preconditions, not another flow encoder.

## Proposed figure 1: three distinct modifications

Given source video (human Pick/lift) and same target final goal:
1. **Replace:** source picks and lifts over physical wall; push-only robot must keep cube supported by table and find an executable pushing path.
2. **Delete:** source makes a habitual, task-irrelevant detour; target does not waste movements imitating it.
3. **Insert:** source picks over a blocked passage; target has no push-only detour, but can first **push a movable blocker aside**, then push goal cube through the cleared passage. Target-only necessary intermediate action is absent from human source. This is a harder and more publishable problem than geometric waypoint remapping.
4. **Reject:** no feasible target behavior under capability constraints; correctly predict unreachable, not fabricate success.

Case 3 must be implemented in real PhysX and compared with a capable task-and-motion planner, not just an A* route. Otherwise the innovation would be hand-coded rearrangement.

## Research hypotheses and decisive counterfactuals

H1. Direct imitation of source object flow becomes systematically infeasible when support, grasp, contact and workspace modes differ. Not all source flows are semantically necessary.

H2. Task effect structure can be decomposed into final target relations, *causal* preconditions, optional source motions, and conditional physical modes. Learn these from TRAIN data and interventions, not test-time privileged-state labels.

H3. A capability-conditioned **effect edit/projection** module can output feasible target events: retain goal and essential prerequisites, delete incidental events, replace infeasible effects, insert new required subgoals. Closed-loop robot policy executes and replans these under observation feedback.

H4. Human source information must be tested against a Goal-only baseline receiving the **same explicit task goal**. If goal+target observation fully specifies all behavior and source videos add no extra useful effect information, the claimed source-to-target transfer problem collapses into ordinary planning. Demonstrate where source demonstrations transmit task-specific relations, constraints or action-effect expectations not given by goal alone, and report cases where they do not.

Counterfactual 2x2 evaluation:
- same source video, target capabilities differ (grasp permitted vs push-only): effect edits should change appropriately while preserving final goal;
- same target/goal, source video changes only in habitual detour: essential target effects and success should remain stable;
- same source/task/capability, target passage/obstacle configuration changes: target should insert/omit clearing effects;
- truly unreachable: calibrated rejection rather than a failed random rollout.

This **target-dependent solution reversal** under otherwise matched data is a necessary experimental signature of capability-aware transfer. Merely repeating the same geometric route under both conditions does not establish novelty.

## Concrete method: Effect Graph -> Capability-conditioned Effect Editor -> Receding-horizon Policy

Observation: source RGB video V_s; target RGB observation o_t with known camera calibration; target control/capability description c_t (e.g. push-only, gripper open, grasp available). Main comparison never uses simulator ground-truth positions, contact labels or obstacle geometry at test-time except as **separately marked oracle upper bounds**.

Representation: E_s = (G, N, \prec, M), with G final object relationships; N observable object-event nodes; \prec causal/temporal prerequisites; M contact/support and allowable object-state modes. Video features via a frozen local pretrained encoder/tracker and lightweight temporal event decoder.

Learned event importance (which source event is necessary) must come from TRAIN interventions/effect-outcome labels and alternate demonstrations: a visited waypoint is not, by itself, a necessary causal event. Learning setup uses simulated target physically realized rollouts and logged failures; test cannot select effects using evaluation reward labels.

Editor P_theta(E_s, o_t, c_t) predicts retain/delete/replace/insert on object-effect event graph, with target dynamics/contact feasibility and goal-consistency scoring. Optimization goal (conceptual):

    E_t^* = argmin_{E} [ L_goal(E,G) + lambda L_necessary(E,E_s)
                         + beta L_infeasible(E,c_t,o_t)
                         + gamma L_extra(E) ]

where L_infeasible must be learned/tested from **measured** transitions rather than arbitrary geometry heuristics. This is not a theorem of certified feasibility.

Policy pi_phi(a_t | o_t,E_t^*) is trained on real/simulated *target robot action-labeled* TRAIN trajectories, **not** on six human videos alone. Receding-horizon re-evaluates completed object events, contact/support state and subsequent feasible effects.

Computation strategy for RTX4090: frozen DINOv2/SigLIP-like local visual encoder, object tracker, small temporal/graph network and ordinary BC/DP downstream policy. Avoid expensive LLM/VLM APIs. Keep same low-level policy backbone and TRAIN action budget for key ablations.

## Experimental evidence: two separate tables, never mix incompatible tasks

**Official RoboReel (general baseline competence):** preserve original, exactly comparable PushCube, PutBasket, ObjectInDrawer, StackCups outcomes, video splits and author evaluator. Original PickCube vs PushCube have *different goals* and must not be misrepresented as a same-task cross-behavior pair. Official Vid2Robot ND PushCube 3/60 (from previous committed actual report) is the only confirmed formal π0.5/Vid2Robot result at present; other official methods remain NOT_RUN/BLOCKED unless their actual per-trial data appear.

**CUSTOM-ADDITIONAL (primary scientific claim):** four task families tested in actual ManiSkill/PhysX:
A) fixed wall / source lift -> push detour; 
B) avoid optional source detour / reduce unnecessary effects; 
C) narrow corridor blocked by a movable object -> first clear then push; 
D) target physically unreachable -> reject.
Include different layouts, obstacle heights and lengths, movable-object mass/friction, alternative target modes, and held-out object/configuration splits. Add a longer multi-effect scene to show transfer isn't confined to single cube translation.

**Strong equal-information controls:** Goal-only (given same explicit goal), raw source object flow, simple time warp, 2D/3D geometry + waypoint, target dynamics-aware task-and-motion planning, flow generation + fixed BC/DP executor, and direct video-conditioned policy. Compare Cross-Behavior Object Flow, Im2Flow2Act, EC-Flow, UniSkill and official video-conditioned pi0.5 on the tasks where genuinely reproducible under matched observations, training and evaluator; label officially provided weights versus ports and reimplementations. Privileged-state oracles have own table, no leakage into RGB leaderboard. VLM API methods deferred by user decision.

**Ablations:** no causal-event importance; no capability condition; no feasibility/dynamics score; no insertion/deletion editor; open-loop vs closed-loop; learned flow vs true source flow (oracle, separate); all else same backbone/steps/data. Add TRAIN size curves and held-out layout/robot-mode splits.

**Metrics:** task success per independent rollout k/N with uncertainty; correct intermediate-effect realization, causal precondition success, edit precision/recall against held-out physical annotations, contact/support violations, collisions, invalid-grasp events, goal completion under mode change, unnecessary action/trajectory cost, unreachable detection false-positive/negative, inference and wall-clock robot steps. Use actual simulator success flags.

## Formal stop/go criteria — avoid a month of unproductive work

STOP the claim early if:
- Same-goal source and target both cannot be physically executed, preventing ground-truth pair construction.
- Source demonstration contains no extra useful information over Goal-only under fair task specification; source transfer is scientifically empty in selected task regime.
- An information-matched 3D geometry/waypoint or task-and-motion planner replicates all effect edits and target physical success, including blocker rearrangement, without meaningful failure cases.
- Target capability changes fail to reverse the selected object-effect sequence or feasibility prediction.
- New method has no gain over raw-flow + same downstream policy and same training budget on unseen paired physical cases.

GO to full experiments only after valid PhysX feasibility, controlled causal counterfactual reversal, learned editor improvement over strong controls, and reliable trained closed-loop policy with genuine robot action labels. A successful scripted teacher is not learned-policy evidence.

## Suggested paper structure

1. Introduction: videos give task effects but may propose physically impossible object motions under target actuation.
2. Related work: flow-as-interface, cross-behavior, embodiment-centric kinematics, task-and-motion/multi-object effect planning.
3. Problem formulation: effect-level task equivalence under heterogeneous behavior/capability constraints; source-video information assumptions.
4. Method: causal object-effect graph, target feasibility-conditioned event edits, receding-horizon closed-loop policy.
5. Benchmarks and reproduction: official RoboReel + transparently labeled additional same-goal PhysX scenes.
6. Experiments: original baselines, capability-switch counterfactuals, target-only effect insertion, unseen generalization, ablations, unreachable rejection and physical robot demonstrations if available.
7. Limitations: uncertain perceptual grounding, no formal feasibility guarantees, data collection and policy mismatch.

Possible title (working only): **Beyond Object Flow Imitation: Capability-Aware Effect Adaptation for Cross-Behavior Robotic Manipulation**.

Central abstract hook: *The same goal can be accomplished through different object-state transitions, but a source demonstration's motion may be outside the target's feasible behavior set; target-side effect adaptation rather than trajectory copying is necessary when constraints alter the causal sequence of physical interaction.*
