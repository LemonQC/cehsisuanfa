# B07 — RoboReel VLM + π0.5 method card

Status: official annotation source and local Qwen3-VL path verified; annotation-only two-video TRAIN DEV runner added. This is not yet a trained/evaluated B07 policy.

## Source lock

- RoboReel model repository: https://github.com/weiweigu1998/roboreel_models
- Pinned revision: `79a0fb0099e73c5560114753a8706276f93aefc2`
- Official annotation entrypoint: `scripts/annotate_human_videos.py`
- Default VLM: `Qwen/Qwen3-VL-4B-Instruct`; loaded locally through Transformers, not an OpenAI/cloud API.
- Official annotation output: `human_video_annotation_<model_name>.json`, keyed by `<video_type>/<task>/<demo>`.

## Bounded first run

`scripts/run_b07_local_qwen_annotation_dev.sh` checks the pinned upstream revision, project-local UniSkill Python, two genuine TRAIN StackCups front videos, free disk and CUDA/Transformers versions. It annotates exactly TRAIN IDs 0000 and 0001, 16 sampled frames per video, using local Qwen3-VL-4B-Instruct and the existing project HF cache. It writes to a unique new JSON filename and refuses to overwrite it.

This only validates the local VLM annotation component. It does not train the VLM+π0.5 policy, run PhysX, or provide a success rate. The next gate is to verify the official B07 checkpoint, its expected annotation format and feature dependencies, then run no more than two TRAIN-only DEV episodes before any held-out EVAL use.

## Why this is not B04

B04 keyframe π0.5 uses `scripts/extract_keyframes.py` and `src/keyframes/vlm_client.py`, which instantiate the OpenAI client and default to GPT-family models. B07's human-video instruction annotation is a separate official path that supports local Qwen3-VL. Do not conflate the two.

## Current gate

`L0_SOURCE=PINNED`; `L1_LOCAL_QWEN_ANNOTATION=DEV_RUNNER_READY`; `POLICY_TRAIN/EVAL=NOT_RUN`; `FORMAL=NOT_RUN`.
