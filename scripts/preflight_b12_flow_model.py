#!/usr/bin/env python3
"""Single-GPU memory preflight for the pinned EC-Flow flow predictor.

Loads only the official DiT-XL-NoPosEmb-Lang flow predictor and optionally
an already-cached OpenAI CLIP RN50. It does not import SAM2/GroundingDINO/
CoTracker, launch Meta-World, or modify upstream source.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ECFLOW = ROOT / "third_party" / "EC-Flow"
CKPT = ECFLOW / "ckpt" / "flow.pt"
CLIP_CACHE = Path.home() / ".cache" / "clip" / "RN50.pt"


def host_available_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def gib(n):
    return round(n / (1024**3), 3) if n is not None else None


def cuda_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "free_gib": gib(int(free)),
        "total_gib": gib(int(total)),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-clip", action="store_true",
                        help="load RN50 only if its local cache already exists")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--min-host-free-gib", type=float, default=12.0)
    parser.add_argument("--out", default="outputs/research_runs/b12_flow_memory_preflight_20261114.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("BLOCKED: CUDA is unavailable")
    if not CKPT.is_file():
        raise SystemExit(f"BLOCKED: missing flow checkpoint: {CKPT}")
    if args.include_clip and not CLIP_CACHE.is_file():
        raise SystemExit(f"BLOCKED: CLIP RN50 cache missing; refusing implicit download: {CLIP_CACHE}")

    host_before = host_available_bytes()
    if host_before is not None and host_before < args.min_host_free_gib * 1024**3:
        raise SystemExit(
            f"BLOCKED: only {gib(host_before)} GiB host RAM available; "
            f"require {args.min_host_free_gib:.1f} GiB"
        )

    sys.path.insert(0, str(ECFLOW))
    from model import DiT_models

    if "DiT-XL-NoPosEmb-Lang" not in DiT_models:
        raise SystemExit("BLOCKED: expected official DiT model key missing")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    model = DiT_models["DiT-XL-NoPosEmb-Lang"]().to("cuda")
    result = {
        "schema": "TRO1_B12_FLOW_MEMORY_PREFLIGHT_V1",
        "checkpoint": str(CKPT),
        "checkpoint_bytes": CKPT.stat().st_size,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": "FP32",
        "batch_size": 1,
        "num_points": 400,
        "prediction_horizon": 8,
        "host_available_before_bytes": host_before,
        "host_available_before_gib": gib(host_before),
        "clip_included": bool(args.include_clip),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in model.parameters()),
        "status": "RUNNING",
    }
    result["model_parameter_gib_fp32"] = gib(result["model_parameter_bytes_fp32"])
    result["after_model_init"] = cuda_snapshot()

    # Match upstream find_model(): load checkpoint on CPU and prefer EMA.
    # weights_only=False is needed for the authors' legacy checkpoint format;
    # only load the checkpoint downloaded from the pinned official model repo.
    checkpoint = torch.load(
        str(CKPT),
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    state_dict = checkpoint["ema"] if isinstance(checkpoint, dict) and "ema" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    del state_dict, checkpoint
    gc.collect()
    model.eval()
    torch.cuda.synchronize()
    result["after_checkpoint_load"] = cuda_snapshot()
    host_after = host_available_bytes()
    result["host_available_after_load_bytes"] = host_after
    result["host_available_after_load_gib"] = gib(host_after)

    # Match official model input dimensions: [B, 3*T, N], RGB image 128x128,
    # language embedding 1024. One forward pass is a bounded memory probe.
    x = torch.randn(1, 24, 400, device="cuda", dtype=torch.float32)
    t = torch.tensor([100], device="cuda", dtype=torch.float32)
    y = torch.randn(1, 3, 128, 128, device="cuda", dtype=torch.float32)
    lang = torch.randn(1, 1024, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = model(x, t, y, lang)
    torch.cuda.synchronize()
    result["forward_output_shape"] = list(out.shape)
    result["after_one_forward"] = cuda_snapshot()

    clip_model = None
    if args.include_clip:
        import clip
        clip_model, _ = clip.load("RN50", device="cuda", download_root=str(CLIP_CACHE.parent))
        clip_model.eval()
        tokens = clip.tokenize(["push the object"]).to("cuda")
        with torch.inference_mode():
            text_embedding = clip_model.encode_text(tokens)
        torch.cuda.synchronize()
        result["clip_output_shape"] = list(text_embedding.shape)
        result["after_clip_load_and_encode"] = cuda_snapshot()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    result["final_free_vram_bytes"] = int(free_bytes)
    result["final_free_vram_gib"] = gib(int(free_bytes))
    result["final_total_vram_bytes"] = int(total_bytes)
    result["peak_allocated_gib"] = gib(int(torch.cuda.max_memory_allocated()))
    result["peak_reserved_gib"] = gib(int(torch.cuda.max_memory_reserved()))
    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["headroom_required_gib"] = args.min_free_gib
    result["status"] = (
        "PASS_SINGLE_GPU_MEMORY_PREFLIGHT"
        if free_bytes >= args.min_free_gib * 1024**3
        else "FAIL_INSUFFICIENT_VRAM_HEADROOM"
    )
    result["scope_note"] = (
        "Model load plus one FP32 forward only; not full 250-step diffusion sampling, "
        "CLIP unless requested, Meta-World rollout, or end-to-end peak."
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE_EXISTING_REPORT: {out_path}")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print("TRO1_B12_MEMORY_PREFLIGHT", json.dumps(result, sort_keys=True))
    print(f"TRO1_B12_MEMORY_PREFLIGHT_SAVED {out_path}")

    if clip_model is not None:
        del clip_model
    del model, out, x, t, y, lang
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if result["status"] == "PASS_SINGLE_GPU_MEMORY_PREFLIGHT" else 10


if __name__ == "__main__":
    raise SystemExit(main())
