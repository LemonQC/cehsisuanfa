# 2026-11-10 — v1.5.3 Official SigLIP cache absent; avoid further 11.6GB GCS attempts

## Observed on RTX4090

Operator updated main to v1.5.2 and ran original local-only cache check.
It listed /home/ubuntu/.cache/openpi and /home/fu/.cache/openpi;
neither contains pi05_vision_encoder/siglip_params.pkl, neither contains
the base pi05_base/params directory. Recursive find of /home/fu,
/home/ubuntu and /mnt/4Tdata/LemonData produced zero siglip_params.pkl.
The downloaded 6.2GB released RoboReel Uniform policy params exist.
JAX GPU was previously available. Zero official pi05 physical trials.

## Why the *next* step is an official DATASET metadata check

The pinned original RoboReel README explicitly describes SigLIP
video_demonstration_features as DERIVED ARTIFACTS from
compute_vision_features_for_demo.py and says they are required at
inference time. It does not prove that the public dataset hosts all
20 already-computed EVAL bundles; check the exact manifest first.

New read-only script:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/run_v141_next_gate.sh audit-siglip-dataset

The script retrieves ONLY public Hugging Face dataset file metadata from
hf-mirror.com (API uses no tokens, no video, no GCS and NO checkpoint
download). It checks the original expected paths:

    video_demonstration_features/without_distraction/push_cube/0080/front.npy
    ...
    video_demonstration_features/without_distraction/push_cube/0099/front.npy

If all 20 are listed, verify their source and schema before selectively
fetching only those files. If not, no need to try another weight mirror in
the next step. Metadata check absence is NOT proof the authors cannot
provide features separately upon request.

## Alternative local method experiment: important provenance boundary

Original SigLipTokenizer is hard-coded to extract:
`pi05_base_params["PaliGemma"]["img"]` and write
`$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`.
Our released finetuned Uniform policy Orbax checkpoint may also contain
a `PaliGemma.img` subtree, but it has NOT yet been inspected, and it
must NOT be assumed numerically identical to the OpenPI original base.
A separately labeled NON-OFFICIAL visual-feature extraction experiment
may use its available image encoder after verifying model architecture.
This is a valid exploratory troubleshooting fork, not a faithful
official RoboReel baseline until parameter equality or original-author
approval/verification is established. Never copy a new finetuned
siglip_params.pkl into the default official cache and silently claim
an original pretrained encoder.

For the faithful original π0.5 track, prefer an authentic author-provided
SigLIP pickle or independently verifiable original pi05_base OpenPI
checkpoint transfer. Current gs://openpi-assets connection has failed
at 0.00/11.6G; do not repeat that download without a working route.

The frozen Vid2Robot result remains 3/60, no replay.
