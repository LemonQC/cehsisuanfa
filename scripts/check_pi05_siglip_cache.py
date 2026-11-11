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


def trusted_mirror_install(project: Path, home: Path) -> bool:
    """Verify previously staged mirror has a REAL cache path below home.

    OpenPI's maybe_download calls .resolve() and then .relative_to(cache_dir).
    An otherwise valid directory SYMLINK pointing outside home will crash.
    All checkpoint files therefore must be available as hardlinks inside
    the physical cache directory (no extra 12.4GB disk copy).
    """
    manifest=project / "outputs/v120_baselines/openpi_base_hf_manifest.json"
    if not manifest.is_file():
        return False
    try:
        record=json.loads(manifest.read_text())
        expected=home/"openpi-assets/checkpoints/pi05_base/params"
        source=Path(record["local_dir"])/"pi05_base/params"
        check=record["manifest_verification"]
        if not (
            record["source"]=="robotgeneralist/openpi_checkpoint_mirrors2"
            and record["revision"]=="1fda6b04e83d625eb1a55d68db32050e248cac2b"
            and Path(record["openpi_data_home"]).resolve()==home.resolve()
            and Path(record.get("openpi_base_cache_path",
                                record["openpi_base_cache_link"])).absolute()==expected.absolute()
            and expected.is_dir() and not expected.is_symlink()
            and expected.resolve().is_relative_to(home.resolve())
            and bool(check["complete"])
            and source.is_dir()
        ):
            return False
        files=[p for p in source.rglob("*") if p.is_file() and not p.name.endswith(".part")]
        return (bool(files)
                and all((expected/p.relative_to(source)).is_file()
                        and os.path.samefile(p,expected/p.relative_to(source))
                        for p in files))
    except (ValueError,KeyError,TypeError,OSError):
        return False


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
    if trusted_mirror_install(args.project.resolve(), root):
        print("PI05_PINNED_HF_BASE_STAGED: original SigLIP tokenizer can "
              "use the existing local cache path without reaching GCS.",flush=True)
        return 0
    if status["base_params_possible"]:
        print("PI05_BASE_CACHE_POSSIBLE: metadata exists but checkpoint completeness "
              "is unverified; inspect before allowing tokenizer restore.", flush=True)
        if not args.permit_gcs_download:
            return 2
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
