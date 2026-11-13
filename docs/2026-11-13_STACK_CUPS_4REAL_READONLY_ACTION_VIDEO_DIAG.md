# TRO1 StackCups official TRAIN-video DEV v1: four real rollouts and post-hoc diagnosis

## Verified results (RTX 4090, user terminal 2026-11-13)

Official RoboReel `StackCups-v1`, two scene states and two same-task genuine human TRAIN videos `0000`, `0001`; frozen `demo_pi05` checkpoint, identical reset states and policy RNG within each paired scene, 800 PhysX steps per rollout.

| Paired episode | Source video IDs (execution order) | Success | Original mean 8D action L2 over common steps |
|---|---|---|---:|
| 0 | 0000, 0001 | False, False | 0.7579299807548523 |
| 1 | 0001, 0000 | False, False | 1.2224546670913696 |

All **4/4 physical rollouts finished at 800 steps without policy runtime errors; 0/4 task completions**. All four `rollout.mp4` paths exist and `video_error=null` in the user's report. This is a successful *benchmark execution*, not successful manipulation. No within-scene binary outcome reversal between video demonstrations, and no evidence of improvement from the video selection. The nonzero full-trajectory mean 8D action difference means trajectories are different, but is not causal proof that the video is useful: closed-loop state divergence can amplify an early perturbation. Arm joints and gripper are combined in that distance, unnormalized. **No CEPS or new recovery method was implemented/tested.** Do not advance T-RO novelty on this evidence.

The JAX / `os.fork()` multithreading warning during MP4 encoding did not prevent completing the rollout; it should be recorded but cannot be identified as the manipulation failure cause.

## Frozen evidence location

The server retains `outputs/roboreel_video_dependency_dev/stack_v1/` containing `results.json`, `SUMMARY.md`, `per_trial.csv`, `episode_*/source_*/actions.npy`, `status.json`, `success_trace.json`, `rollout.mp4`. No uploaded raw data were available to the assistant; all numerical observations above come from the user's terminal output. **Do not rerun or overwrite `stack_v1`.** No PartNet-Mobility required, no EVAL 0080–0099 used.

## Read-only diagnostic tool

`python -m scripts.diagnose_stack_video_dev` operates ONLY on existing completed evidence and explicitly validates all action-file SHA256 against the original trial manifest, expected task, four trials, same initial RNG, and the originally reported full-trajectory L2. It never imports JAX/OpenPI/ManiSkill, calls `env.step`, downloads data, or changes the evidence root. Outputs go into a new directory `outputs/roboreel_video_dependency_diagnosis/stack_v1`:

- `diagnosis.json`: matched pairwise action differences at 1, 20, 50, 100, 200, 400, 800 steps, separated into 7-DOF arm and gripper; first action delta and reference magnitude.
- `SUMMARY.md`: compact comparison for discussion.
- `rollout_montage.jpg`: four rows of 10 timestamps decoded from the original robot rollout MP4s. The montage is only a visual reference; it does not automatically assign manipulation-stage successes or failures.

Test fixture: `tests/test_diagnose_stack_video_dev.py`, 8 offline `unittest` tests. Tested locally using NumPy fixtures; no real user's server videos/actions were inspected by assistant. If the montage can't be generated due to imageio/ffmpeg/Pillow, `diagnosis.json` still records the exact error and metrics; original MP4 files remain intact.

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
conda activate pi05
git pull --ff-only origin main
python -m unittest discover -s tests -p 'test_diagnose_stack_video_dev.py' -v
python -m scripts.diagnose_stack_video_dev
cat outputs/roboreel_video_dependency_diagnosis/stack_v1/SUMMARY.md
```

The user should send the printed `ROBOREEL_STACK_READONLY_DIAGNOSIS` line and attach `rollout_montage.jpg` to enable a grounded visual classification of reaching/grasping/first stack/second stack stages. Do not invent a failure stage from `success=false` alone. If the montage reveals no meaningful long-horizon stage transition or first-action differences are negligible, stop and report `NO-GO`; do not spend further GPU trials without a new falsifiable hypothesis and a same-information baseline.
