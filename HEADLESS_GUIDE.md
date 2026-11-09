> **v0.4.0 update (2026-11-09):** The MPLib crash was bypassed in v0.3.0. The first successful physical-motion pilot reached only ~half of the first requested lateral waypoint on both sides. The expert now performs up to two *measured* re-contacts per missed waypoint by default; `PUSH_ATTEMPT` logs the start/error/end/progress. Follow the current [README.md](README.md) and use a fresh `outputs/ee_v040_episodes` folder. No revised success rate is claimed until server results arrive.\n\n# v0.3.0 diagnosis — native MPLib crash

Latest 4090 trace:
- mplib/planner.py, line 65, __init__
- mani_skill/examples/motionplanning/panda/motionplanner.py
- agcofl/teacher.py, scripted_rollout

This is not the previous GLX/Vulkan error. Python exception handling cannot catch the native crash. We removed the MPLib dependency in teacher data collection and use ManiSkill's pd_ee_delta_pos (XYZ delta + gripper) built-in IK controller.

Run:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    unset DISPLAY WAYLAND_DISPLAY
    export MPLBACKEND=Agg PYTHONFAULTHANDLER=1
    python -m pytest -q tests
    python -u -X faulthandler -m agcofl.cli doctor 2>&1 | tee doctor_v030.log
    python -u -X faulthandler -m agcofl.cli collect --episodes 1 --seed 100 --out outputs/ee_v030_episodes 2>&1 | tee collect_v030.log
    cat outputs/ee_v030_episodes/collection_summary.json

If it still crashes, share doctor_v030.log and collect_v030.log. If it no longer crashes but the teacher fails, share outputs/ee_v030_episodes/route_*.json, which now records actual physical failure details. Do not train on failed episodes.

The changed controller means old episodes and checkpoints are incompatible. --video only draws post-simulation 2D trajectory MP4; no Vulkan is used.
