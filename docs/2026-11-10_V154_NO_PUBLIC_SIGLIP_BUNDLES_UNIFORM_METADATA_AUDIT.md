# 2026-11-10 — v1.5.4 Official precomputed SigLIP bundles absent; inspect downloaded fine-tuned checkpoint

## Confirmed evidence, not an inference

The operator executed the metadata-only official dataset audit using
hf-mirror.com: repository LogosRoboticsLab/roboreel, revision
846f5dfdde5d7269fd1cda90bde8264c5cac0e29, listed
`eval_front_bundles_found=0`, `derived_feature_file_count=0`,
missing all EVAL front.npy IDs 0080–0099. No model, video, feature or
checkpoint bytes were downloaded by that probe.

Earlier local search found no SigLIP pickle under either default OpenPI
cache or a recursive search in /home/fu, /home/ubuntu and
/mnt/4Tdata/LemonData. Previous attempt to download
gs://openpi-assets/checkpoints/pi05_base/params (11.6GB)
stalled at 0B and was interrupted. No original official π0.5 physical
evaluation has been executed.

The original pinned author's `SigLipTokenizer` restores
`pi05_base` and extracts its `PaliGemma.img` parameter subtree.
The released fine-tuned RoboReel Uniform 6.2GB policy checkpoint may
also contain a compatible `PaliGemma.img` subtree, but this has not
been observed or verified. Even if present, **the fine-tuned image
encoder is not assumed bit-identical to original OpenPI base**.
Using it for SigLIP video preprocessing without matching parameter
hashes or author validation would be a *modified baseline*, not a
faithful reproduction.

## New read-only checkpoint inspection

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/run_v141_next_gate.sh inspect-uniform-vision

This reads only local checkpoint `params/_METADATA` (up to 32MiB)
and its `train_config.json`. It does NOT load tensor arrays,
deserialize pickles, allocate a JAX GPU model, or call Hugging Face/GCS.
An Orbax `tree_metadata` matching key is **an indicator only**; metadata
layouts vary, so absence of a candidate match is NOT conclusive
evidence that no visual subtree exists. The original image encoder
and fine-tuned counterpart's parameter equality is not established.

## Scientific next action

1. If the metadata lists `PaliGemma.img`, inspect parameter shapes
   and compare exact identity to an authentic `pi05_base` image
   subtree (e.g. author-supplied standalone SigLIP cache). Do not
   declare official baseline equivalence without source verification.
2. If authentic base visual parameters are unavailable, label any
   fine-tuned-vision feature adapter `NON_OFFICIAL` and isolate its
   outputs from the original RoboReel official pi05 success table.
3. In parallel, proceed to the same-goal Pick→Push PhysX physical
   feasibility DEV experiment under `agcofl4090`, avoiding further
   download deadlocks. Do not confuse scripted privileged physics
   with a trainable new method.

Official Vid2Robot 3/60 remains frozen, no reruns. The author-release
π0.5 Uniform policy cannot be marked EVALUATED until authentic visual
conditioning is available and actual 3×20 physical rollout results
are verified.
