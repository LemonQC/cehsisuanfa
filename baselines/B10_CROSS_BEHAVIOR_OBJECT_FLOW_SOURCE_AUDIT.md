# B10 — Cross-Behavior Object Flow source audit

Status: paper and accepted manuscript verified; no official public author code/checkpoint repository was verified in the sources checked on 2026-11-14.

## Paper lock

- Title: “Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation”.
- Authors: Longrui Chen, David Russell, Yulei Qiu, Mehmet Remzi Dogar.
- Venue: IEEE Transactions on Robotics, 2026, volume 42, pages 2631–2647.
- DOI: 10.1109/TRO.2026.3701576.
- Accepted manuscript: https://eprints.whiterose.ac.uk/id/eprint/242108/ (CC-BY 4.0 author-produced version).

## Reproduction status

The paper's central object-flow representation is directly relevant to this project's cross-behavior topic. The public sources located provide the paper and accepted manuscript, but the repository search did not identify a verified official code/weight release. Therefore B10 remains `BLOCKED_AUTHOR_CODE_UNVERIFIED`, not an implemented or reproduced baseline.

## Next action

1. Search the accepted manuscript/supplementary material for implementation/data pointers.
2. If no code is linked, contact the corresponding author for code, dataset access and license.
3. If unavailable, consider a paper-faithful reimplementation only after deriving the exact model, losses, data construction and training protocol from the manuscript. Record each deviation; do not label it official.
4. Keep B10 out of any success-rate table until the original task/data and a faithful execution are verified.

## Current gate

`L0_PAPER=VERIFIED`; `L0_OFFICIAL_CODE=NOT_VERIFIED`; `NATIVE=NOT_RUN`; `ROBOREEL_PORT=NOT_STARTED`; `FORMAL=NOT_RUN`.
