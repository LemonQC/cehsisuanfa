# AG-COFL Partial Object-Flow Supervision — ManiSkill 3 Pilot

Version v0.3.0. FIX: the prior crash in mplib/planner.py:65 inside PandaArmMotionPlanningSolver has been bypassed. The new agcofl/teacher.py uses ManiSkill's built-in pd_ee_delta_pos Cartesian controller (XYZ+gripper) for all real Panda physics actions. It NEVER instantiates MPLib.

The official PushCube doctor, custom task, teacher, BC, and Gate A now use the same four-dimensional pd_ee_delta_pos controller. Existing v0.2 pd_joint_pos data and checkpoints must not be mixed; the dataset loader checks its controller metadata.

Physics success of the new teacher is NOT yet verified on the user's RTX 4090. This code is a candidate experimental rollout, not claimed successful data. Stop if the robot cannot push reliably through both routes. Do not train on failures.

## Run on existing 4090 conda env (NO reinstall)

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    unset DISPLAY WAYLAND_DISPLAY
    export MPLBACKEND=Agg PYTHONFAULTHANDLER=1

    python -m pytest -q tests
    python -m agcofl.cli test-loss
    python -u -X faulthandler -m agcofl.cli doctor 2>&1 | tee doctor_v030.log

After STOCK_PHYSICS_OK and SMOKE_OK:

    python -u -X faulthandler -m agcofl.cli collect --episodes 1 --seed 100 --out outputs/ee_v030_episodes 2>&1 | tee collect_v030.log
    cat outputs/ee_v030_episodes/collection_summary.json

Do not use --video on the first run. If you later add it, it makes only a headless 2D animation from measured cube XY positions, NOT RGB camera video. The renderer remains disabled.

Normal failure messages such as Failed TCP hover/descend, Cube missed waypoint, or Failed contact push mean the IK/physical demonstration geometry needs correction. They are NOT native segfaults; inspect saved per-episode JSON, and do NOT reinstall CUDA or NVIDIA drivers.

Only after there are successful recorded physical demonstrations for both sides:

    python -m agcofl.cli collect --episodes 15 --seed 200 --out outputs/ee_v030_episodes
    python -m agcofl.cli train-bc --data outputs/ee_v030_episodes --epochs 100 --ckpt outputs/models/flow_bc_ee_v030.pt
    python -m agcofl.cli gate-a --ckpt outputs/models/flow_bc_ee_v030.pt --episodes 5 --seed 9000 --max-steps 500 --video

The goal of Gate A is to demonstrate that same-initial-state Panda physically selects upper vs lower tabletop route in response only to the synthetic future object-mask sequence. This is a feasibility experiment, NOT an already-proven result. Full/Anchor/End losses are implemented but the entire high-level human-to-robot training comparison is NOT yet executed.

Official ManiSkill controller documentation: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html

See HEADLESS_GUIDE.md for diagnosis.
