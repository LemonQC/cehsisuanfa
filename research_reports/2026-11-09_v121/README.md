# v1.2.1 — Released Vid2Robot checkpoint and official evaluator recovery

v1.2 already obtained and validated **two authentic TRAIN RoboReel robot
H5 datasets**, with 138x8 and 136x8 policy actions and real base/hand camera
MP4s. Those files stay local and are NOT downloaded again in this stage.

Previous v1.2 blocker: HF metadata detected a ~2.37GB original released
Vid2Robot checkpoint and 20 original evaluation videos (~97.45MB). The
original HF download attempt failed with LocalEntryNotFoundError. This stage:
- explicitly binds hf_hub_download to the official or mirror endpoint;
- reuses any local files that match the PUBLISHED exact byte size;
- reuses the earlier official EVAL video 0080 only if exact size matches;
- retries and resumes HTTP downloads from verified HF resolve hosts;
- writes per-file verified local SHA256 without sharing original bytes;
- runs only the authors' original Vid2Robot/ManiSkill runner, never a
  substituted policy or privileged-state scripted controller.
- DEV-only single heldout 0080 rollout is a pipeline check, NOT a paper
  score; no tuning/model selection is allowed on EVAL demonstrations.

No official success can be claimed unless the JSON says
rollout_executed=true and a measured official_success flag is present.
Any contact or robot dynamics result requires actual physics rollout.
