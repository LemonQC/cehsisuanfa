# v1.1 Baseline-first / pretrained ViT report

- Official baseline model code: https://github.com/weiweigu1998/roboreel_models
- Frozen reference commit: 79a0fb0099e73c5560114753a8706276f93aefc2
- Methods: video-conditioned pi0.5 uniform + keyframe, Vid2Robot, UniSkill, VLM+pi0.5.
- An official source entrypoint is NOT an executed benchmark baseline; check
  checkpoint_loaded and official_rollout_executed fields before any claim.
- ViT report is pretrained DINOv2-S ViT on real RoboReel TRAIN videos with
  video-level split, camera-plane 2D pseudo-centers, and a temporal Transformer.
- ViT diagnostic is NOT a learned control policy, a 3D effect extractor,
  a semantic segmentation QA pass or an official RoboReel success rate.
- Large videos, action H5s, model checkpoints and private env files stay on host.
- The T-RO hypothesis is NOT rejected by the earlier weak CNN pilot.
