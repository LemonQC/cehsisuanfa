# B04 — π0.5 Keyframe method card

Status: official keyframe policy/checkpoint configuration exists; official keyframe annotation stage requires an OpenAI API client, and no existing authorized per-demo keyframe annotation bundle has yet been verified locally.

## Source lock

- RoboReel model repository: https://github.com/weiweigu1998/roboreel_models
- Pinned revision: `79a0fb0099e73c5560114753a8706276f93aefc2`
- Policy evaluation config: `configs/eval/nd/vid_cond_pi05_keyframes.json`
- Official annotation entrypoint: `scripts/extract_keyframes.py`
- API client: `src/keyframes/vlm_client.py`

## Official pipeline and constraint

The official extractor samples front-camera frames and asks a VLM to return JSON keyframe indices. The pinned client imports `OpenAI`, constructs `OpenAI()` and defaults to GPT-4o-mini (the CLI config defaults to GPT-5.4-mini). Thus generating missing annotations through the original method requires an external OpenAI API key and is blocked by the project's no-cloud-API constraint.

## Allowed next steps

1. Read-only audit for pre-existing `bpp_keyframe_annotations/<bucket>/<task>/<demo>.json` under the already downloaded data roots. If all required TRAIN and evaluation annotations are already present and licensed, use them without calling a VLM.
2. If missing, optionally implement a separate `B04_OFFLINE_KEYFRAME_REIMPLEMENTATION` using local hand-motion/keypoint speed or temporal change. Keep its annotation JSON, model slug and result rows separate from the official B04 method. Do not call it official keyframe reproduction.
3. Before evaluation, verify the exact annotation schema and the checkpoint's expected data lookup path.

## Current gate

`L0_SOURCE=PINNED`; `OFFICIAL_ANNOTATION=BLOCKED_API_UNLESS_PREANNOTATED`; `LOCAL_REIMPLEMENTATION=NOT_STARTED`; `POLICY_EVAL=NOT_RUN`; `FORMAL=NOT_RUN`.
