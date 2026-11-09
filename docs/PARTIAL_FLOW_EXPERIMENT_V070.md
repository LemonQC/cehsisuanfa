# Partial Object-Flow: actual experiment plan (v0.7.0)

The core pilot compares three supervision regimes on *exactly the same* clips,
network, split and training budget:

- Full: robot Full + human Full.
- Anchor: robot Full + human midpoint-of-arclength anchor plus stage endpoint.
- End: robot Full + human stage endpoint only.

Robot and human losses are averaged separately; human weight is fixed 0.5.
The split is by unique layout ID (not frames, episodes, or repeated seeds).
Training seeds can change while the held-out split stays fixed via --split-seed.

## Immediately runnable, on the existing RTX 4090 Conda environment

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    bash scripts/run_partial_study_pilot.sh

This command runs pytest and the existing mask loss check, generates explicitly
SYNTHETIC geometrical proxy masks, trains Full/Anchor/End high-level predictors,
and produces CSV/JSON/PNG figures.

This is **not an actual T-RO experimental result**. There are no human video
frames, no robotic policies, and no robot actions in this pipeline smoke.

Outputs in outputs/proxy_full_anchor_end_v070/:

- report.json: contains SYNTHETIC_PROXY_DEBUG_NOT_PAPER warning and setup
- split.json: exactly which layout groups enter train/val/test
- full.pt, anchor.pt, end.pt: trained high-level model checkpoints
- training_history.csv: loss of each mode (different objective; interpret cautiously)
- predictions.csv: per segment IoU on validation and held-out test
- robot_test_summary.csv: mean mask IoU grouped by condition and mode
- paired_differences.csv: per-matched-robot-clip DeltaIoU from Full to Anchor/End
- offline_comparison.png, training_curves.png: diagnostic plots

## Real data needed for the paper

The importer accepts JSONL manifests of binary masks with the same coordinates
and definition across robot/human clips. Every stage must have eight float32
binary masks, shape (8,32,32). Example JSONL record:

    {"id":"layout1_human_phase0","layout_id":"layout1",
     "condition":"useful","phase":0,"source":"human",
     "mask_path":"masks/layout1_human_phase0.npy",
     "context":[-0.145,0.0,0.0,0.155,0,1,0.155,0.0,1]}

The context holds observable stage start, *known task-specified event goal*,
stage and scene attributes. Never construct this model input from unseen
future labels or the target supervision mode. Identical source clips and
context must be reused for all modes. Same task boundary event is applied
to all source clips, and full source mask sequences are retained even when
training only a subset.

A second importer turns **successfully executed** ManiSkill robot NPZ files
and **real annotated, calibrated human world-XY trajectories** into two
segment clips. The latter is a projected silhouette proxy, NOT real camera
segmentation and not comparable directly to the original RGB-based model.
Use native real video mask clips in the direct manifest for a faithful study.

Input: research_recordings.jsonl in repo root. Fields:
id, layout_id, source (robot/human), condition, path, check_goal,
final_goal, scene. For human additionally boundary_index marking
the observed first arrival at the inspection region. For robot use the
existing NPZ file recorded from real env.step() and success=True.
Failed robot demonstrations are skipped by design.

Example robot line (adjust path to your existing files):

    {"id":"L001_robot","layout_id":"L001","source":"robot",
     "condition":"useful",
     "path":"outputs/ee_v050_episodes/route_upper_000100.npz",
     "check_goal":[0,0.155],"final_goal":[0.145,0],
     "scene":[1,0.155,0,1]}

Example human line after recording the human video and calibrating XY:

    {"id":"L001_human","layout_id":"L001","source":"human",
     "condition":"useful","path":"raw/human_01_world_xy.npy",
     "boundary_index":74,"check_goal":[0,0.155],
     "final_goal":[0.145,0],"scene":[1,0.155,0,1]}

To prepare, train and visualize:

    python -m agcofl.prepare_real_segments \
      --index research_recordings.jsonl \
      --out datasets/real_mask_segments

    python -m agcofl.partial_flow_study train \
      --manifest datasets/real_mask_segments/manifest.jsonl \
      --out outputs/real_study_seed42 \
      --epochs 100 --seed 42 --split-seed 42 --device auto

    python -m agcofl.plot_partial_study --out outputs/real_study_seed42

Run seeds 43 and 44 with identical --split-seed 42 and different output folders.
The training pipeline REJECTS fewer than six independent layout IDs.
The existing static two-route environment has only ONE layout; different
seed values alone are not scene generalization.

## Data that must be collected, not invented

1. Multiple distinct inspected tabletop layouts, with reproducible geometry,
   timestamps, obstacle positions, camera calibration and layout IDs.
2. Robot demos, initially Push-only, then Pick-capable. Full state/actions,
   true cube XY/XYZ, RGB+depth+binary object masks, checkpoint visits, success,
   slip/collision/support constraint events. Keep failures separately.
3. Real HUMAN videos paired by layout/task, with useful alignment and
   feasible-but-habitual detour categories; both source and target can
   finish the task but middle route differs. Same human clips in all modes.
4. Stage boundaries at first inspection event, independent held-out layouts
   and source-video IDs; no adjacent frames split across datasets.
5. CLOSED-LOOP downstream evaluation on the same starting states under
   each trained high-level predictor's PREDICTED masks (not oracle labels).
   Report ordered task success, actual route and collision, efficiency,
   training seeds, layout-paired confidence intervals and runtime.

The current project provides an offline supervision intervention and an
existing but not fully validated low-level BC gate. It does NOT yet establish
human-to-robot transfer, or the claimed T-RO novelty. The core paper
criterion remains that physically executable human motion can be
unhelpful supervision, and that this affects robotic closed-loop results.
