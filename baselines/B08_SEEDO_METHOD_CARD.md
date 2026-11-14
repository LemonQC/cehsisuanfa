# B08 — SeeDo method card

Status: public author repository and task assets located; the official VLM stage requires OpenAI GPT-4o API, so the full original pipeline is blocked under the project's no-paid/no-cloud-API constraint.

## Source lock

- Repository: https://github.com/ai4ce/SeeDo
- Revision: `48ff18827c626c196648dc92333491c9e9dd8948` (main, 2026-07-23)
- Paper: IROS 2025, arXiv:2410.08792.
- The README links a public Hugging Face dataset containing a SeeDo assets.zip for supported PyBullet objects.

## Native pipeline

1. Hand-motion keyframe selection with MediaPipe (`get_frame_by_hands.py`).
2. Object visual prompting/tracking with GroundingDINO, SAM and SAM2 (`track_objects.py`).
3. VLM reasoning in `vlm.py`.
4. PyBullet pick-and-place execution through a modified Code-as-Policies stack.

The repository includes an automated CSV plan-evaluation script. The simulator is only executable for object models and action primitives already implemented in its prompt/asset library; unseen objects require new models and functions.

## Hard API dependency

The pinned README instructs users to create `VLM_CaP/src/key.py` with an OpenAI API key. The pinned `vlm.py` imports `OpenAI`, creates an API client and hard-codes `gpt-4o`. Therefore the original official full SeeDo method cannot be run under the current no-external-API rule.

An offline Qwen3-VL replacement can be built as `B08_LOCAL_VLM_REIMPLEMENTATION`, but it must be labeled as a reimplementation and must not be reported as official SeeDo. This replacement also needs to preserve prompts/output schema and independently evaluate plan accuracy; it does not automatically make the original PyBullet primitive executor compatible with RoboReel.

## Current gate

`L0_SOURCE=PUBLIC_PINNED`; `KEYFRAME/PERCEPTION=CAN_BE_AUDITED`; `OFFICIAL_VLM=L0_BLOCKED_API`; `LOCAL_QWEN_REIMPLEMENTATION=NOT_STARTED`; `ROBOREEL_PORT=NOT_STARTED`; `FORMAL=NOT_RUN`.
