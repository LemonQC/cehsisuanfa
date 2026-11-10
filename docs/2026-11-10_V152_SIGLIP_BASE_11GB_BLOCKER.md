# 2026-11-10 — v1.5.2 SigLIP preprocessing attempts to download a second 11.6GB π0.5 base checkpoint

## Operator facts: why 0% for 94 seconds

The user ran the real first-video SigLIP smoke in `pi05`. The GPU
import and NumPy ABI gates had been addressed, but the original author's
`SigLipTokenizer.__init__` invoked:

    openpi.shared.download.maybe_download(
        "gs://openpi-assets/checkpoints/pi05_base/params")

The logs show gcsfs "Your default credentials were not found" warnings,
falling back to GCSFileSystem, progress 0.00/11.6G for >94 s,
and user KeyboardInterrupt. No significant model bytes were confirmed
transferred, no official eval rollout executed. Missing ADC credentials
do **not** by themselves mean a public GCS object requires paid auth;
the remote connection or GCS listing may be blocked/unreliable. Do not
recommend configuring cloud billing credentials as a workaround.

This **separate OpenPI BASE** checkpoint is NOT the 6.20GB released
RoboReel ND Uniform fine-tuned policy checkpoint already downloaded.
The pinned author source extracts only:
`pi05_params["PaliGemma"]["img"]` to
`$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`.
That pickle is then reused for all EVAL video feature extraction.
There is no evidence that a vision encoder taken from the fine-tuned
6.20GB checkpoint is equivalent to the base encoder. Therefore
substituting it silently would violate exact official baseline fidelity.

## New no-download checks

New code `scripts/check_pi05_siglip_cache.py` audits paths:
- `$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl`,
- default `~/.cache/openpi/pi05_vision_encoder/siglip_params.pkl`,
- likely caches under /home/ubuntu or /home/fu;
- any partial GCS base `openpi-assets/checkpoints/pi05_base/params`;
- released fine-tuned Uniform checkpoint (presence only, not equivalence).

The `smoke-one-feature` and `prepare-features` entrypoints now fail
BEFORE invoking original SigLipTokenizer when SigLIP cache is unavailable,
so no accidental second 11.6GB GCS download. A user can intentionally
request original cloud access with `PI05_ALLOW_GCS_DOWNLOAD=1`, but it
will still require working GCS networking.

On the 4090:

    cd /mnt/4Tdata/LemonData/cehsisuanfa
    git pull --ff-only origin main
    conda activate pi05
    bash scripts/run_v141_next_gate.sh check-siglip-cache

For read-only broader local search (do NOT print/unpickle strangers' files):

    find /home/fu /home/ubuntu /mnt/4Tdata/LemonData \
      -type f -name siglip_params.pkl -printf '%p %s bytes\n' 2>/dev/null

If an authentic pickle is found at a different location, set
`OPENPI_DATA_HOME` to its *parent cache root*, or use a verified copy
in the expected `pi05_vision_encoder/siglip_params.pkl` location.
Do not copy untrusted pickles; Python pickle can execute arbitrary code.
Checkpoint identity and source must be established.

## Realistic fallback, prioritized

1. Check exact shared cache first. The source will not download if a valid
   source-authenticated pickle already exists.
2. Check whether the *official RoboReel dataset* offers the original
   bundled `video_demonstration_features/.../front.npy` for the 20 EVAL
   IDs. A legitimate precomputed cache produced by the **same original
   SigLIP encoder/checkpoint** would avoid any local 11.6GB base download.
   Do not rename a different vision model's embeddings as SigLIP.
   Dataset feature availability has NOT been confirmed; do not promise it.
3. If neither exists, download the original base checkpoint via a
   working authorized network/GCS transfer on another machine and only
   then export the official SigLIP img subtree once. The full 11.6GB
   transfer may still be required because of Orbax sharding; don't
   claim a simple file-level image-only HTTP download is possible.
4. A separately declared **NON-OFFICIAL feature adapter experiment** could
   test extracting from fine-tuned weights, but must be clearly labeled
   as different and cannot replace the official π0.5 Uniform results.
   Its actual compatibility is untested.

Only after original `siglip_params.pkl` exists and one-video smoke proves
actual tensor loading should `prepare-features`, `--require-ready` and
`evaluate-uniform` be considered. v1.3 Vid2Robot 3/60 remains frozen.
No π0.5 official physics trials have run.
