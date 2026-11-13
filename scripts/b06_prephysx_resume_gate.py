#!/usr/bin/env python3
"""Verify exact B06 DEV pre-PhysX state before reusing an output directory.

Do not delete/archive previous results. The only recoverable state is a
completed immutable manifest + preflight, with NO episode/trial data. An
unknown file, symlink, altered hash or partial trial is an irreversible STOP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SCHEMA = "TRO1_B06_UNISKILL_STACKCUPS_DEV_V1"
PREFLIGHT = "TRO1_B06_UNISKILL_STACKCUPS_PREFLIGHT_V1"
SAFE_FILENAMES = frozenset({"manifest.json", "preflight.json"})
HEX256 = re.compile(r"^[0-9a-f]{64}$")


def check_unstarted_run(out: Path, project: Path, plan: dict | None = None) -> dict:
    out = Path(out).absolute()
    project = Path(project).resolve()
    root = project / "outputs" / "research_runs"
    if (out.parent != root or out.name != "b06_uniskill_stackcups_dev_v1"
            or out.is_symlink() or not out.is_dir()):
        raise ValueError("B06_RESUME_UNEXPECTED_OUTPUT_DIRECTORY")
    if any(p.is_symlink() or not p.is_file() for p in out.iterdir()):
        raise ValueError("B06_RESUME_SYMLINK_OR_NONFILE_BLOCKED")
    names = {p.name for p in out.iterdir()}
    if names != SAFE_FILENAMES:
        raise ValueError("B06_RESUME_PARTIAL_OR_UNKNOWN_FILES: " + repr(sorted(names)))
    raw = {}
    for name in sorted(SAFE_FILENAMES):
        p = out / name
        if p.stat().st_size > 4 * 1024 * 1024 or p.stat().st_size < 30:
            raise ValueError("B06_RESUME_INVALID_JSON_SIZE: " + name)
        raw[name] = json.loads(p.read_text(encoding="utf-8"))
    m, pre = raw["manifest.json"], raw["preflight.json"]
    if (m.get("schema") != SCHEMA or pre.get("schema") != PREFLIGHT
            or m.get("status") != "RUNNING_BOUNDED_DEV_NOT_FORMAL"
            or pre.get("status") != "READY_FOR_DEV"
            or m.get("task") != "stack_cups" or pre.get("task") != "stack_cups"
            or m.get("episodes") != 2 or m.get("max_steps") != 800
            or m.get("source_train_demo_ids") != [0, 1]
            or pre.get("source_train_demo_ids") != [0, 1]
            or m.get("train_videos_only") is not True
            or m.get("no_heldout_eval_videos") is not True
            or m.get("no_policy_training") is not True
            or pre.get("new_physx_actions") != 0):
        raise ValueError("B06_RESUME_MANIFEST_OR_PREFLIGHT_MISMATCH")
    base = pre.get("base_checks", {})
    expected = {
        "policy_checkpoint_sha256": "policy_checkpoint_sha256",
        "idm_checkpoint_sha256": "idm_checkpoint_sha256",
        "official_upstream_revision": "upstream_revision",
    }
    for mkey, pkey in expected.items():
        value = m.get(mkey)
        if not isinstance(value, str) or value != base.get(pkey):
            raise ValueError("B06_RESUME_PINNED_ASSET_MISMATCH: " + mkey)
    for key in ("policy_checkpoint_sha256", "idm_checkpoint_sha256"):
        if not HEX256.fullmatch(m[key]):
            raise ValueError("B06_RESUME_INVALID_SHA256: " + key)
    for manifest_key, rows_key in (("source_video_sha256", "train_videos"),
                                   ("skill_sha256", "skill_arrays")):
        hashes = m.get(manifest_key, {})
        rows = base.get(rows_key, [])
        if (set(hashes) != {"0", "1"} or not isinstance(rows, list)
                or len(rows) != 2):
            raise ValueError("B06_RESUME_DEMO_PROVENANCE_INCOMPLETE")
        for row in rows:
            if hashes.get(str(row.get("id"))) != row.get("sha256"):
                raise ValueError("B06_RESUME_DEMO_HASH_MISMATCH")
    if plan is not None:
        if plan.get("status") != "READY_FOR_DEV" or plan.get("task") != "stack_cups":
            raise ValueError("B06_RESUME_CURRENT_PLAN_NOT_READY")
        current = plan.get("base_checks", {})
        for manifest_key, plan_key in expected.items():
            if m.get(manifest_key) != current.get(plan_key):
                raise ValueError("B06_RESUME_CURRENT_ASSET_CHANGED: " + manifest_key)
        for manifest_key, rows_key in (("source_video_sha256", "train_videos"),
                                       ("skill_sha256", "skill_arrays")):
            now = {str(row["id"]): row.get("sha256") for row in current.get(rows_key, [])}
            if m.get(manifest_key) != now:
                raise ValueError("B06_RESUME_CURRENT_DEMO_CHANGED")
    print("B06_PREPHYSX_RESUME_VERIFIED metadata_only=2 prior_actions=0 "+
          "existing_output_preserved=true", flush=True)
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        check_unstarted_run(args.out, args.project)
    except Exception as exc:
        print("B06_PREPHYSX_RESUME_BLOCKED type=" + type(exc).__name__ +
              " reason=" + str(exc)[:220], flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
