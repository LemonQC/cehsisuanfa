#!/usr/bin/env python3
"""Read-only audit of the SigLIP source-weight dependency, NO network.

Original pinned RoboReel SigLipTokenizer reads
$OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl and otherwise
downloads gs://openpi-assets/checkpoints/pi05_base/params (~11.6GB).
The released ND Uniform policy checkpoint is NOT automatically equivalent.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def audit(project: Path, home: Path):
    candidates = []
    for base in (
        home,
        Path("~/.cache/openpi").expanduser(),
        Path("/home/ubuntu/.cache/openpi"),
        Path("/home/fu/.cache/openpi"),
    ):
        if base not in candidates:
            candidates.append(base)
    checks = []
    cached_pickle = False
    cached_base = False
    for base in candidates:
        pickle = base / "pi05_vision_encoder" / "siglip_params.pkl"
        params = base / "openpi-assets" / "checkpoints" / "pi05_base" / "params"
        row = {
            "cache_root": str(base),
            "siglip_pickle": str(pickle),
            "siglip_pickle_bytes": pickle.stat().st_size if pickle.is_file() else None,
            "base_params": str(params),
            "base_params_directory_exists": params.is_dir(),
            "base_params_metadata_present": (params / "_METADATA").is_file()
        }
        cached_pickle |= bool(row["siglip_pickle_bytes"])
        cached_base |= bool(row["base_params_directory_exists"] and
                            row["base_params_metadata_present"])
        checks.append(row)
    released = project / "outputs/v120_baselines/released_checkpoints/nd/vid_cond_pi05_uniform/ckpt"
    summary = {
        "type": "PI05_SIGLIP_NO_DOWNLOAD_PREFLIGHT",
        "cache_roots": checks,
        "siglip_pickle_available": cached_pickle,
        "base_params_possible": cached_base,
        "uniform_policy_params_present": (released / "params").is_dir(),
        "uniform_is_not_proven_equivalent_to_base": True,
        "no_cloud_access_attempted": True,
    }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--permit-gcs-download", action="store_true",
                   help="Explicitly acknowledge optional 11.6GB GCS transfer; this tool itself never downloads")
    args = p.parse_args(argv)
    root = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    status = audit(args.project.resolve(), root)
    print("PI05_SIGLIP_DEPENDENCY", json.dumps(status, ensure_ascii=False), flush=True)
    if status["siglip_pickle_available"]:
        print("SIGLIP_CACHE_PRESENT: run one-video smoke to test actual restore.", flush=True)
        return 0
    if status["base_params_possible"]:
        print("PI05_BASE_CACHE_PRESENT: original tokenizer can export the SigLIP pickle; "
              "run one-video smoke only if base params are complete.", flush=True)
        return 0
    if args.permit_gcs_download:
        print("SIGLIP_BASE_DOWNLOAD_EXPLICITLY_ALLOWED: this audit did not download anything.",
              flush=True)
        return 0
    print("PI05_SIGLIP_BLOCKED_NO_CACHED_VISION_WEIGHTS: original tokenizer "
          "will attempt gs://openpi-assets/checkpoints/pi05_base/params (~11.6GB). "
          "Do not run smoke-one-feature until original vision params are available.",
          flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
