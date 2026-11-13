# 2026-11-13 | ObjectInDrawer frozen success-trace horizon audit

## Validated server evidence so far

The user ran four **actual PhysX** `toys_in_drawer` trials after authorized PartNet-Mobility 44962 installation. GitHub evidence sources:
- Real runs: https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_540ff3963863f5ed/evidence.json
- Read-only action diagnosis: https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/evidence/runs/drawer_hf_v1_40a7bc0821b412be/evidence.json

Observed: 2/4 success in the **800-step custom DEV** (Episode 1 trials successful after 634 and 650 steps), no matched-pair success reversal (0/2); Episode 0 fails in both demos after 800 steps. Matched first-20 action L2 values ~0.001443 / ~0.001332; full shared-trajectory mean 1.340733 / 0.270444. These are descriptive action sensitivity, not proof of video-dependent goal information.

Original `ObjectInDrawer-v1` ManiSkill registration defaults to **400 episode steps**, and original `make_eval_env` supports explicit override. A success after 634/650 steps cannot be presented as a success under 400-step default. Importantly, **the stored `success_trace.json` was not included in the first sanitized reports**; it may indicate transient successes earlier than the termination step. Only reading that existing trace can decide exactly whether any success occurred within the 400-step prefix.

## Implemented next-step code (no new PhysX)

- `scripts/audit_drawer_success_horizon.py`: validates frozen `results.json` (two paired scenes/four source trials), verifies every original 8D action file SHA256 matches its trial record, reads four `success_trace.json` booleans, checks trace lengths and seed matching, computes exact first successful step, and reports `success_by_400` vs `success_by_800` without speculation. Stops on mismatched/fabricated/symlinked evidence. Uses no simulator, model, network or human data.
- `tests/test_audit_drawer_success_horizon.py`: ten CPU-only tests, run successfully in an isolated local fixture, including 400 vs 800, transient success, action tampering, matched seeds, original file immutability and refusal to overwrite.
- `scripts/diagnose_drawer_horizon_and_publish.sh`: immutable one-time local audit into `outputs/roboreel_video_dependency_diagnosis/drawer_horizon_v1`, followed by GitHub sanitized evidence publisher to the *separate* `tro1-experiment-evidence` branch.
- Existing `scripts/publish_research_evidence.py` now includes only the **numeric, validated horizon summary** from `horizon_audit.json`, not raw action arrays, source human videos, unfiltered local paths, license files, or secrets.

### Run once on research server

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_audit_drawer_success_horizon.py' -v
bash scripts/diagnose_drawer_horizon_and_publish.sh
```

If the audit reports 0/4 success by 400, it establishes an **offline DEV-prefix** failure of this particular frozen policy and seeds; it does not establish official benchmark accuracy, because it is not the official heldout evaluation bank or experiment scope. If success appears before 400, report precisely which trial/step; do not infer it just from the final episode length. **No additional GPU runs** are authorized by this audit.

## Research decision

**NO-GO for building a Stage-Sync / CEPS algorithm from the current fixed-goal two-demo evidence.** The two source human videos encode the same nominal drawer task; within matched scenes they give the same success outcome. Initial actions differ by ~1e-3 and the longer-horizon trajectories diverge; the causal source-video utility is not isolated. The new authorized PartNet assets make a workable multistage *evaluation platform*, not a novel algorithm claim.

Keep RoboReel π0.5, Vid2Robot, object-in-drawer scripts, PartNet provenance, benchmark manifests and frozen negative evidence. A new algorithm must first be justified by a genuine target/contact constraint counterfactual with real human-source evidence and strong *same-information* baselines. Until that exists, stop adding seeds/architectures or training runs to this hypothesis.
