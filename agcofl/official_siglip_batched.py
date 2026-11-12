"""Memory-bounded adapter for the PINNED RoboReel SigLIP feature encoder.

Only execution is batched. Frame preprocessing, the original SigLIP callable,
spatial pooling, positional embeddings, and final bundle schema are exactly
those of upstream _PersistentEncoder.encode_demo_camera(). Do NOT use to
manufacture or reorder video demonstrations.

The original author method sends all T video frames to JAX at once and can
allocate multi-GiB resize intermediates. This module changes only axis-0
batching; no checkpoint, source video or task semantics change.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

IMAGE_GRIDS = (("8x8", 64), ("4x4", 16), ("2x2", 4))


def validate_batch_frames(n: int) -> int:
    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError('SIGLIP_BATCH_FRAMES must be in 1..8, not a full-video batch')
    return n


def encode_camera_batched(encoder, out_dir: Path, rgb: np.ndarray,
                          camera: str = 'front', *, batch_frames: int = 2,
                          progress=None) -> str:
    """Write an author-schema bundled .npy with short JAX frame batches.

    The supplied encoder MUST be the unmodified, pinned author's
    _PersistentEncoder, with the actual pre-trained SigLIP weights. This method
    is not a replacement model. The same operations:

      normalize float32 -> resize_with_pad(224,224) -> _siglip_call
      -> pool_tokens_to_size(64,16,4) -> device_get

    is applied to consecutive slices rather than one [T,...] tensor.
    """
    validate_batch_frames(batch_frames)
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError('Expected decoded video [T,H,W,3] numpy RGB frames')
    if rgb.dtype != np.uint8:
        raise ValueError('Expected unmodified genuine uint8 RGB video frames')
    t_total = int(rgb.shape[0])
    if t_total < 2 or t_total > int(encoder._max_T):
        raise ValueError(f'Video length T={t_total} outside 2..{encoder._max_T}')
    if camera != 'front':
        # Train-only device plan intentionally conditions on the one validated
        # front camera; this path must not quietly infer other views.
        raise ValueError('This bounded TRAIN run supports only front camera')

    out_dir = Path(out_dir)
    bundle_path = out_dir / f'{camera}.npy'
    if bundle_path.exists():
        raise FileExistsError('Refuse to overwrite existing SigLIP bundle: ' + str(bundle_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = {name: [] for name, _ in IMAGE_GRIDS}
    # The pinned original processes every frame in chronological order. Keep
    # each operation and normalization order byte-for-byte equivalent in NumPy.
    for begin in range(0, t_total, batch_frames):
        end = min(begin + batch_frames, t_total)
        part = (rgb[begin:end].astype(np.float32) / 255.0) * 2.0 - 1.0
        x = encoder._jnp.array(part[:, None, ...])
        flat = encoder._einops.rearrange(x, 'k v h w c -> (k v) h w c')
        resized = encoder._image_tools.resize_with_pad(flat, 224, 224)
        batched = encoder._einops.rearrange(
            resized, '(k v) h w c -> k v h w c', k=end - begin, v=1)
        tokens = encoder._siglip_call(batched)
        # Each pool sees the original 256-patch token grid and makes the exact
        # same (T,64/16/4,2048) embedding, just a shorter leading dimension.
        for grid, target_tokens in IMAGE_GRIDS:
            pooled = np.asarray(encoder._jax.device_get(encoder._pool(tokens, target_tokens)))
            if pooled.ndim != 4 or pooled.shape[0] != end - begin or pooled.shape[1] != 1:
                raise RuntimeError(f'Unexpected official pooled token shape for {grid}: {pooled.shape}')
            if pooled.shape[2] != target_tokens or pooled.shape[3] != 2048:
                raise RuntimeError(f'Incorrect official token grid/dim for {grid}: {pooled.shape}')
            collected[grid].append(pooled[:, 0])
        del tokens, batched, resized, flat, x, part
        if progress:
            progress(end, t_total)

    bundle = {}
    for grid, target_tokens in IMAGE_GRIDS:
        stacked = np.concatenate(collected[grid], axis=0)
        if stacked.shape != (t_total, target_tokens, 2048):
            raise RuntimeError('Bundled SigLIP image embeddings have invalid shape')
        bundle['image_emb_' + grid] = stacked
        positions = np.asarray(encoder._pe[grid][:t_total])
        if positions.shape != (t_total, target_tokens, 768):
            raise RuntimeError('Pinned official positional embedding shape differs: '+str(positions.shape))
        bundle['pos_emb_' + grid] = positions
    bundle['state_emb'] = np.zeros((t_total, 8), dtype=np.float32)
    bundle['timesteps'] = np.arange(t_total, dtype=np.int32)
    encoder._atomic_save(bundle_path, bundle)
    if not bundle_path.is_file() or bundle_path.stat().st_size < 512:
        raise RuntimeError('Failed to write official-schema feature bundle')
    return f'wrote author-schema batched bundle T={t_total} batch_frames={batch_frames}'
