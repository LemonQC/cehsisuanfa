#!/usr/bin/env python3
"""Read-only preflight for the author's separate OpenPI/JAX pi0.5 environment.

Run *after activating the target Conda environment*. No packages installed,
no weights downloaded, and no physical rollouts launched.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = PROJECT / "third_party" / "roboreel_models"

FEATURE_IMPORTS = (
    "jax",
    "jax.numpy",
    "numpy",
    "imageio.v2",
    "einops",
    "openpi.shared.image_tools",
    "demonstration_vla_suite.shared.siglip_tokenizer",
    "demonstration_vla_suite.shared.posemb_3d",
    "demonstration_vla_suite.shared.data_utils",
)
EVAL_IMPORTS = (
    "openpi.models.model",
    "openpi.transforms",
    "openpi.shared.nnx_utils",
    "demonstration_vla_suite.models.lfo_pi0",
    "demonstration_vla_suite.models.lfo_observation",
    "demonstration_vla_suite.policies.transforms",
    "gymnasium",
    "mani_skill",
    "sapien",
    "orbax.checkpoint",
)


def import_roots(official: Path = OFFICIAL) -> list[str]:
    return [str(official / p) for p in (
        "", "src", "src/external_models",
        "src/external_models/lerobot/src", "roboreel_env/src")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("features", "eval"), required=True)
    args = parser.parse_args(argv)
    for root in reversed(import_roots()):
        if root not in sys.path:
            sys.path.insert(0, root)
    checks = {}
    if not OFFICIAL.is_dir():
        checks["official_source"] = "MISSING"
    modules = FEATURE_IMPORTS + (EVAL_IMPORTS if args.mode == "eval" else ())
    for name in modules:
        try:
            importlib.import_module(name)
            checks[name] = "OK"
        except Exception as exc:
            checks[name] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:170]
    devices = []
    if checks.get("jax") == "OK":
        try:
            import jax
            devices = [f"{d.platform}:{d.id}" for d in jax.devices()]
        except Exception as exc:
            checks["jax_devices"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:170]
    gpu_ok = any(device.startswith("gpu:") or device.startswith("cuda:")
                 for device in devices)
    summary = {"type": "PI05_RUNTIME_READ_ONLY", "python": sys.executable,
               "mode": args.mode, "jax_devices": devices,
               "gpu_visible": gpu_ok, "checks": checks,
               "siglip_weights_restored": "NOT_TESTED",
               "physical_rollouts": 0,
               "ready_for_next_gate": all(x == "OK" for x in checks.values()) and gpu_ok}
    print("PI05_RUNTIME_CHECK", json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["ready_for_next_gate"]:
        print("PI05_RUNTIME_BLOCKED: use the author's ENVIRONMENTS.md section 1 "
              "in a dedicated JAX/OpenPI Conda environment; do not modify agcofl4090",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
