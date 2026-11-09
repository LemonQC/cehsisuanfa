# v1.2.1 verified → v1.3 official Vid2Robot full evaluation (2026-11-09)

## Verified facts from GitHub auto-upload ede31ff

- Official Vid2Robot released ND checkpoint 2,368,873,705 bytes plus train_config.json 911 bytes are present and size/SHA256-verified.
- Official PushCube no-distraction EVAL front videos 0080–0099: 20 out of 20 available, total 97,448,129 bytes.
- The pretrained original Vid2Robot model was STRICTLY loaded and drove the authors' original RoboReel ManiSkill physical environment with camera inputs and no privileged-state policy.
- Fixed episode 0080 actually ran, returned success=false (0/1). This is only a 200-step DEV integration check, not paper performance, and is not an argument against the research direction.
- Previous 2 real PushCube TRAIN robot action H5s: 138×8, 136×8, plus real base/hand MP4s already validated and kept local.

## Locked full baseline protocol, no EVAL model tuning

Now run the first measured official learning-based policy baseline. Do NOT run more 2D-ViT regressions or resample new smoke episodes.

- Source model code: weiweigu1998/roboreel_models at commit 79a0fb0099e73c5560114753a8706276f93aefc2.
- Model checkpoint: LogosRoboticsLab/roboreel_models, nd/vid2robot/ckpt.pt, original pretrained weights. No fine-tuning.
- Environment: original RoboReel ManiSkill with panda_wristcam_hd.
- Task: PushCube, **one clean human/clean environment ND cell only**, not 10 tasks and not WD/ED.
- Held-out human video IDs: exactly 0080–0099 (20 unique demos) in each pass.
- Observations: actual human front RGB + robot base RGB camera history.
- Original controller: pd_joint_pos with chunk_size=1 and max_rollout_steps=800.
- Original environment passes: seed_offset_base 0, 100000, 200000 (three physical state-bank passes, 20 examples each).
- Accept each pass ONLY after checking exact 20 unique source IDs, original success Boolean flags, the original evaluator's aggregate table, steps and metadata.
- Report original success counts per pass, mean ± SEM across three seeds, AND successes/trials (60 total when complete). Do not treat repeated videos over 3 seeded states as 60 independent human demonstrations.
- The only implementation delta is suppressing a redundant timm ImageNet initialization download which would otherwise be overwritten by STRICT full released Vid2Robot state_dict. All actual model parameters, forward pass, original robot dynamics, control and task success are unchanged.
- One prior EVAL 0080 video was used only to verify the inference/physics integration, and no model parameter or hyperparameter selection occurred based on it. No further tuning on EVAL.

Files added:
agcofl/vid2robot_official_full_eval.py
agcofl/official_v130_protocol.py
tests/test_v130_official_protocol.py
scripts/run_v130_official_vid2robot_3pass.sh
scripts/publish_v130_official_eval.sh

The script reuses existing verified assets without transferring 2.37 GB again, and it resumes at the seed-pass level only if the previous full 20-trial original summary is valid. Machine-readable results auto-upload to research_reports/2026-11-09_v130.

## Research / T-RO baseline and novelty decision

Table A: Official video-conditioned robot policy baselines: released Vid2Robot (first), Vid.Cond.π0.5 Uniform, Vid.Cond.π0.5 Keyframes, UniSkill, VLM+π0.5. Then robot-only ViT-Transformer BC and diffusion policy trained on authentic original robot actions. Other flow baselines will be added only if original source code and compatible experiment conditions can be reproduced. **No fictional success scores**.

Table B: ADDITIONAL physically matched source/target behavior task, separate from official RoboReel. Original PickCube-v2 asks to LIFT and PushCube-v2 asks to PUSH to a goal; that mismatch cannot establish same-goal Pick→Push transfer. Define a fixed-goal task where the human source can Pick-and-Place but the target robot is Push-only; compare flow reuse, geometry-only filters, Goal-only, Goal+Waypoint Oracle, actionability-aware projection and the same-backbone learned robot policy. Include realizable and unrealizable goals; distinguish necessary intermediate effect from irrelevant detour. Measure actual physical success, contact/support violations, preservation of necessary events and task length. Ability-aware projection must beat strong matched-information controls in physical execution, not just produce an A* trajectory.

**Never stop this research solely because an individual old CNN/ViT implementation or one published baseline scores poorly.** Reconsider the specific novelty claim only once actual paired physical evaluations with strong matched controls demonstrate no gap.

## Execute on RTX 4090

cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull --rebase --autostash origin main
conda activate agcofl4090
set -o pipefail
bash scripts/run_v130_official_vid2robot_3pass.sh 2>&1 | tee baseline_v130.log

If and only if GitHub reports full_3pass_complete=true, the clean-cell 3-pass result is ready to be cited as our reproduced official Vid2Robot baseline. All other cases remain partial. Commit was made to the source repository but these rollouts have not yet been executed on the user's server.
