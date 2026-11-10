# 2026-11-10 — v1.4.2: API-free T-RO research priority

## Decision
No paid external LLM/VLM API inference or API-generated keyframes/language annotations in this phase. They do not directly test the scientific claim that source object effects must be projected onto physically executable target behavior.

- B04 official video-conditioned π0.5 Keyframes: DEFERRED, NOT_RUN. Its original annotation generation relies on a VLM API when original keyframe annotations are absent. Do not invent heuristic keyframes and still call results official.
- B07 VLM+π0.5: DEFERRED, NOT_RUN. No Qwen/VLM annotation services.
- B03 official video-conditioned π0.5 Uniform: P0, NOT_RUN. Original JAX/OpenPI inference and local SigLIP video feature extraction; released weights are large but do not require a paid VLM API.
- B05 official Vid2Robot: 3/60 in PushCube ND, already completed v1.3 and frozen.
- UniSkill / robot-only Transformer BC / Diffusion Policy: API-free learning controls, future priority, NOT_RUN in current unified official setting.

## Main scientific gate
Same cube, same seeded state, identical final XY goal. Source Pick/Lift may cross an obstacle but target must Push-only with tabletop support.

Required paired physical cases: source lift feasible/target push detour feasible; necessary intermediate event; irrelevant source-specific detour; genuinely push-unreachable target. Measure actual object trajectories, contact/support constraint violations, reachability errors, task success and robot steps. Compare against raw source flow, Goal-only, waypoint oracle and geometry/simple filtering; later add matched-information learned RGB policies and a trainable capability-aware effect projection policy. Report privileged controls separately from RGB-only policies.

The scripted current custom environment is a physical *feasibility DEV harness*, not a learned T-RO contribution. It has not been run in the RTX4090 server by this commit. Reject any invalid source grasp/release/overflight before comparing targets.

## Repository change
scripts/run_v141_next_gate.sh now exposes ONLY gate, prepare-uniform, evaluate-uniform, custom-physics-dev. agcofl/v141_pi05_assets.py rejects Keyframes and contains no API-extraction entrypoint. Tests protect this restriction. No v1.3 reruns and no new physical results are claimed.

From the server after checking a clean main branch:
    conda activate agcofl4090
    bash scripts/run_v141_next_gate.sh gate
    bash scripts/run_v141_next_gate.sh custom-physics-dev

For Uniform, use original authors' *independent* pi05 environment first:
    bash scripts/run_v141_next_gate.sh prepare-uniform
    bash scripts/run_v141_next_gate.sh evaluate-uniform

Source records: research_reports/2026-11-09_v130/ and research_reports/2026-11-09_v141/status.json. Any future results require actual physical per-trial records, not script completion flags.
