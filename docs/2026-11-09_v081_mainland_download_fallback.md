# Mainland-China RoboReel Video Access — v0.8.1

## 2026-11-09 server result

Previous v0.8.0 run on the RTX4090:

    3 passed in 0.11s
    Agent panda_wristcam_hd is already registered. Skip registration.
    Override registered env PutBasket-v1

    Returning existing local_dir ... as remote repo cannot be accessed
    Downloading 20 videos from LogosRoboticsLab/roboreel...
    20 of 20 videos weren't found ...
    .../push_cube/0080/front.mp4

Interpretation: **video download failed due to official HF endpoint timeout**.
This log says NOTHING about real physical RoboReel benchmark success or
failure; simulation evaluation never began. The ManiSkill registration
warnings are not the fatal error. Do not reinstall the working environment.

## Change in v0.8.1

1. Avoid the upstream 20-video downloader. Use **one** authentic dataset file:
   human_demos/eval/no_distraction/push_cube/0080/front.mp4
2. Default the Python Hugging Face client to the *unofficial*
   https://hf-mirror.com public mirror (or an explicitly given HF_ENDPOINT).
   Set endpoint **before importing** huggingface_hub.
3. Bound the download attempt to 110 seconds and prevent fallback to an
   invalid existing MP4. Mirror availability is NOT guaranteed on a given
   mainland server.
4. Reuse a previously copied authentic file at exactly the same path.
5. If this MP4 is unavailable, skip video segmentation and execute a
   **separately labeled physics-only** check on the public RoboReel saved
   scene. It uses no human video or learned visual policy.
6. Never call the 3-model synthetic study or claim Full/Anchor/End success.
   With a real MP4, validate the mask overlay before applying any video timing.

## Run on the current 4090

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull origin main
    conda activate agcofl4090
    unset DISPLAY WAYLAND_DISPLAY
    export MPLBACKEND=Agg
    set -o pipefail
    bash scripts/run_roboreel_bridge.sh 2>&1 | tee roboreel_cn_v081.log

No driver, CUDA, PyTorch or ManiSkill reinstall.

Manual one-file mirror attempt if needed:

    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
      python -m agcofl.download_roboreel_cn \
        --root third_party/roboreel --demo-id 80 --camera front

If that exact authentic MP4 is provided locally, put it here:

    third_party/roboreel/human_demos/eval/no_distraction/push_cube/0080/front.mp4

The script will use it without network. **Do not create fake/synthetic
video in that path**; recorded evidence must retain real provenance.

## What to send back after execution

If mirror works:
- roboreel_cn_v081.log
- outputs/roboreel_cn/inspect_0080/human_video_diagnostic.json
- outputs/roboreel_cn/inspect_0080/human_mask_overlay.png
- outputs/roboreel_cn/fixed/report.json
- outputs/roboreel_cn/video_motion/report.json (if quality gate passed)
- outputs/roboreel_cn/paired_diagnostic.csv (if both modes passed)

If mirror fails:
- roboreel_cn_v081.log
- outputs/roboreel_cn/physics_only/report.json, IF official saved state
  actually ran successfully. If not, the first meaningful exception line
  in the log will diagnose the upstream ManiSkill 3.0.0b22 / 3.0.1 mismatch.

**Scientific boundary:** this is still a data/physics interface diagnostic
under privileged state-control, NOT human-to-robot transfer. A proper
Full/Anchor/End closed-loop comparison remains pending. Avoid further
hand-tuned teacher code until this external benchmark integration is
verified.

## External source notes

The external HF-Mirror project publicly documents:

    export HF_ENDPOINT=https://hf-mirror.com

as its dataset/model mirror configuration. It is third-party and may have
outages, stale cache, or missing content. There is no guarantee that this
specific RoboReel MP4 will download on the user's network.

The upstream RoboReel repo is https://github.com/gupta-anmol99/roboreel
and specifies the dataset id LogosRoboticsLab/roboreel and the evaluation
video hierarchy.
