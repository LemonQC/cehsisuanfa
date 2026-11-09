# 2026-11-09 v1.3 Run: progress visibility + RTX4090 utilization

## Actual measurements supplied by user

- seed offset 0: OFFICIAL_V130_PASS_VERIFIED, 1 successful physical rollout / 20, 5%; elapsed 332 seconds.
- seed offset 100000: OFFICIAL_V130_PASS_VERIFIED, 2 / 20, 10%; elapsed 319 seconds.
- seed offset 200000: already started and original released Vid2Robot checkpoint STRICTLY loaded; number of current episodes within this third group is NOT known from the original logs.
- Two full groups: 40/60 physical episodes completed, 3/40 successes (7.5%). These are preliminary, **not** the final 3-seed mean or official whole RoboReel benchmark outcome.
- ManiSkill warnings about previously registered panda agent and overriding PutBasket are not fatal in the shown output.

## Root cause: why console seemed stuck

In the upstream RoboReel original evaluation/utils.py run_eval_protocol, the authors DO log task/demo ID, success and duration once per episode, but they do so with logger.info. The project's own launch adapter did not enable root INFO logging, so those messages were invisible. Our orchestration printed markers only at the beginning and end of each entire 20-demo group.

## What changed (observability ONLY)

- New agcofl/official_live_progress.py wraps the original runner's load_demo_static and rollout instance methods without modifying calls/inputs/return values/learned weights/actions/physics.
- Subsequent invocations print OFFICIAL_V130_TRIAL_START and OFFICIAL_V130_TRIAL_DONE per real demo, including ID, sequence number, success flag and measured elapsed time.
- Atomic progress telemetry saved under outputs/v130_official_vid2robot/seed_offset_<seed>/progress.json and append-only progress.jsonl. These files are informative only: final scientific results still use the AUTHORS' ORIGINAL summary.json strictly verified for 20 unique EVAL 0080-0099 records.
- New tests/test_v130_live_progress.py verifies unchanged original-call semantics and no fabricated success on exceptions.
- New scripts/show_v130_progress.sh is read-only; safe from another terminal and also reports nvidia-smi memory and GPU compute utilization.
- This GitHub code cannot modify an ALREADY RUNNING Python process. **DO NOT interrupt the current 3rd seed pass or git pull into a working tree while its process remains active.** After it finishes, the next evaluated task/process will have per-demo progress.

## VRAM 63% vs GPU compute utilization 63%

If 63% refers to **VRAM capacity occupied**, 100% is NOT a useful target; memory usage is not throughput, filling it risks OOM.
If it refers to **GPU-Util**, the sampled fraction of busy GPU compute may vary because the simulator's CPU scene reset, video decode, environment step and GPU model inference alternate. Single-env full-closed-loop chunk_size=1 does not keep the GPU continuously saturated.

Use the separate nvidia-smi metrics. Do NOT modify original chunk_size=1, max_rollout_steps=800 or success criteria just to fill the GPU; then the benchmark would no longer be comparable. Potential speedup from independent processes requires reproducibility and resource isolation tests, which have NOT been performed.

## Inspect safely from a second terminal now

~~~bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
tail -n 15 outputs/v130_official_vid2robot/seed_offset_200000/run.log
~~~

The existing old process may have no episode-level prints. This is a logging limitation, NOT evidence that it has stalled.

After the job has exited and reported V130_AUTOPUBLISH_OK, sync:
~~~bash
git pull --rebase --autostash origin main
bash scripts/show_v130_progress.sh
~~~

If research_reports/2026-11-09_v130/official_aggregated_summary.json has full_3pass_complete=true and three verified physical passes, use it as the clean PushCube official baseline. Do not tune or select the policy based on held-out EVAL 0080–0099. Proceed to additional official learning baselines and separate same-goal capability-mismatch science experiments.
