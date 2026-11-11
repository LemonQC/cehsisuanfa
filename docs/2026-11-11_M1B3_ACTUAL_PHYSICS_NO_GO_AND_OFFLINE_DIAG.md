# 2026-11-11 — M1-B3 actual four-way PhysX result: blocked/open contrast but feasible pair NOT achieved

## 1. Real operator evidence (NOT simulated by this document)

The operator executed the frozen `scripts/run_m1b3_corridor_once.sh`
on the RTX 4090 server and supplied `M1B3_PHYSICS_GATE`, after all
four `RecordEpisode` cases had been generated.

| M1-B3 case | Actual physical result | Steps | Final task XY error |
|---|---|---:|---:|
| blocked_source | **FAIL**; `RuntimeError: official_mplib_lower_failed` | 335 | 0.2100023113 m |
| blocked_direct | **FAIL** (scripted direct pushing) | 91 | 0.1420676067 m |
| blocked_clear | **FAIL**; blocker **physically parked**, task cube not delivered | 450 | 0.0901588300 m |
| open_direct | **SUCCESS**, goal under 20mm tolerance | 35 | 0.0143144273 m |

True reported gate flags:
```
geometry_preflight_pass=true
matched_task_start_and_goal=true
source_pick_place_success=false
blocked_direct_push_success=false
blocked_clear_then_push_success=false
open_direct_push_success=true
blocked_direct_observed_blocker_displacement_m=0.0170175623
blocked_direct_observed_side_clearance=false
clear_side_pocket_observed=true
pilot_contrast_pass=false
causal_necessity_proven=false
strong_tamp_evaluated=false
learned_method_evaluated=false
source_video_information_value_measured=false
paper_ready=false
```

The Gymnasium Tensor-vs-NumPy warnings are not the proximate cause;
all cases reached actual PhysX result reporting. The four local H5,
videos and physical traces should remain at
`outputs/m1b3_corridor_dev/` and **must not be overwritten**.
This repository summary is transcribed from the operator's terminal
log. Original H5, MP4 and 450-step trace bytes have not been
independently fetched from the server and are not claimed to be
present in this GitHub report.

## 2. Interpretation and the technical debugging boundary

**Geometry:** the offline conservative static blocker C-space gate
passed for this shape, but it does not prove universal infeasibility
of dynamic pushing or a true multi-body contact problem.

**Physics feasibility:** open passage + direct Push succeeds.
The blocked direct *single controller* fails and only displaces
blocker about 17.0mm, which is insufficient for side clearance.
The explicit side-clear case moves blocker to parking, but its
subsequent target-cube delivery fails at the 450 action budget.
The source original-OBB/MPLib-adapted Pick+Place is unsuccessful
at the `lower` stage (335 steps, final goal error 210mm).
The existing same-goal source and the blocked target therefore
have **no valid successful matched-goal pair for this scene**.

**Prohibited overclaims:** no learned method was evaluated, no
source video information advantage was established, no
capability-aware equal-information TAMP was run, and the manually
selected clear-first oracle script is not TAMP. A failing
heuristic/direct controller is NOT a proof that any target policy
must insert the clearance step. Under this current pair,
**M1-B3 candidate is NO-GO for method development**.

## 3. One-time evidence-preserving offline diagnosis, not another simulation

New `scripts/m1b3_diagnose_existing.py` and
`tests/test_m1b3_diagnose_existing.py` have been committed.
The script uses Python **standard library only**, reads the existing
four `status.json` + `physical_trace.json`, rejects malformed
trace lengths, and calculates:

- for `blocked_source`: actual saved `planner_phases`,
  attempted `lower`, final and closest target error, real cube
  lift height/grasp counts, and TCP/object movement;
- for `blocked_clear`: independently measured `clear_blocker`
  and `push_cube` phase lengths, intermediate cube/blocker pose,
  observed motion versus stalls and budget flags;
- for `blocked_direct` / `open_direct`: object and TCP displacement,
  nearest goal distance and blocker displacement;
- every input status/trace file's SHA256 for traceability.

The script writes two SMALL reports in the same local output folder:
`diagnosis.json` and `DIAGNOSIS.md`. They are immutable (a
mismatching existing report is not overwritten). It does NOT
import ManiSkill, access GPU, download files, train, or launch
one single new PhysX rollout.

### Next server commands

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --ff-only origin main
conda activate pi05

python -m unittest discover -s tests -p 'test_m1b3_diagnose_existing.py' -q
python -m scripts.m1b3_diagnose_existing
cat outputs/m1b3_corridor_dev/DIAGNOSIS.md
```

The `M1B3_OFFLINE_DIAG` marker and `DIAGNOSIS.md` will
pinpoint whether the last Push stage had insufficient progress,
the robot/TCP could not reach the narrow channel, or the
source lower stage failed before any actual placement.
**Do not claim a specific mechanism before examining real trace data.**
Even an observed stalled push needs collision/pose evidence before
calling it physically impossible.

### Decision after diagnosis

Freeze this four-case dataset (including its failures). Do not
retrain models, increase the seed count, rerun the same scenarios,
or hand-tune parameters in place. A future *versioned* v1 scene
or planner implementation change is allowed only if the
existing measurements establish what failed and a strong
baseline/feasibility experiment can meaningfully distinguish
the proposed capability-editing problem. If it cannot, stop
this causal insertion formulation promptly rather than
spending another month implementing variants.

Historical frozen results remain M0 original π0.5 Uniform 46/60,
Vid2Robot 3/60, ManiSkill original Pick reference 3/3, and
M1-B2 same-goal source/direct/clear all 3 succeed but necessary
insertion contrast fails. None are new learned cross-behavior results.
