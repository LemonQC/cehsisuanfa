# 2026-11-09 · v0.7 Full/Anchor/End 4090 smoke — actual run record

## Observed terminal output (provided by the experimenter)

- `5 passed in 0.62s` (new targeted test suite).
- `LOSS_CHECK_OK` (Full direct-gradient frames 0–7; Anchor [2,7]; End [7]).
- `PROXY_ONLY`: manifest with **96 synthetic masks**, **12 synthetic layouts**.
- `STUDY_COMPLETE`: CUDA, seed 42, split seed 42, epochs 15, human weight 0.5, split **train 8 / validation 2 / test 2** layout IDs.
- `PLOTS_WRITTEN`: `offline_comparison.png`, `training_curves.png`.
- Results are explicitly `SYNTHETIC_PROXY_DEBUG_NOT_PAPER`, no real robot actions.

## Full-trajectory robot-proxy test Mask IoU (each cell n=4 segments)

| Proxy condition | Full | Anchor | End |
|---|---:|---:|---:|
| Habitual detour | 0.1554008 | 0.1762761 | 0.2007624 |
| Useful intermediate route | 0.3100321 | 0.2934508 | 0.3035122 |

Additional observations:
- End-frame IoU for habitual-detour condition is identical (0.3653846) across all modes.
- Differences in the Full trajectory metric are not proof of path compliance, supervision value, cross-behavior transfer, or robot task success.
- The proxy encodes synthetic condition and goal in the feature vector; it is a functional engineering test, **not** a causal research result.

## Decision: GO for infrastructure, NOT YET a scientific hypothesis pass

**Frozen:** Core Full/Anchor/End loss code, paired train/validation/test layout split, training run structure, JSON/CSV output formats, current checked CUDA environment.

**STOP:** Additional synthetic proxy epochs/seeds, hand-tuning contact offsets as the central research program, nine-stage table search, T-RO novelty claims without human/robot policy evidence.

## Next mandatory work: credible end-to-end data + closed-loop

1. Reuse a public *matched* human-video + robot simulation dataset where possible to avoid spending additional weeks making a custom scripted pushing teacher. The 2026 RoboReel benchmark (https://roboreel.github.io/) provides real human demo videos, paired ManiSkill digital-twin tasks, fixed initialization and public code/data (https://github.com/gupta-anmol99/roboreel); start by checking a small PushCube download, **not** the full dataset or expensive VLA.
2. Inspect whether any RoboReel tasks actually exhibit **same goal but different useful/dispensable intermediate paths**. If not, RoboReel may be a secondary generalization benchmark only; it cannot replace the science-critical useful vs habitual route intervention.
3. Replace the v0.7 toy context-only high-level MLP with a fixed predictor that consumes available *observations* (real camera masks or frames, current object state, task goals) rather than a synthetic condition indicator. Keep network, source data, budget fixed across modes.
4. Build and verify one real task's low-level action interface that conditions on predicted object masks (not privileged Oracle route identifiers). The current high-level model predicts 8 masks; earlier FlowBC expects 4-channel synthetic masks. A tested adapter, distribution-shift check, and independent robot evaluation are needed.
5. Collect/create actual matched human-video and robot data with specific stage boundaries and layouts for **useful intermediate route** and **physically executable habitual detour**. Include proof of physical feasibility for the latter. Hold out full layout IDs/video IDs, never adjacent frames across splits.
6. On those actual recordings train Full/Anchor/End using identical architecture, data, random seeds and evaluation. Primary outcome: ordered physical task success, route satisfaction, collision, task time and paired-layout confidence intervals. Offline IoU and masked losses are **secondary diagnostics**.

## Immediate implementation exit criteria

- Next milestone is **one real human video + its correctly aligned robot-simulation episode and object-mask sequence**, producing a real end-to-end policy evaluation trace; not another table of proxy IoUs.
- Before claiming a promising T-RO gap, verify that Full/Anchor/End changes actual robot outcomes on unseen layouts, and that the result withstands a time-aligned Full/waypoint and source-aware Full baseline.

These are decisions for next code integration; no actual new dataset has been downloaded or evaluated in this report.
