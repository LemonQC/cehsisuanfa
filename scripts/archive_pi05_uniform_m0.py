#!/usr/bin/env python3
"""Freeze genuine author-evaluated RoboReel π0.5 Uniform ND PushCube results.

This is OFFLINE, READ-ONLY on original run outputs. Validates all three
actual official 20-trial author summaries using the pinned wrapper's
verify() and writes a small, auditable 60-trial archive with source SHA256.
Does not run simulation, modify weights, download anything, or train.

Run after ACTUAL_PI05_PHYSICS_VERIFIED, from project root:
  python -m scripts.archive_pi05_uniform_m0
  # Direct file invocation is also supported after the module-path fix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys

# Support both invocation forms without asking the operator to modify
# PYTHONPATH or install any unrelated package in the pinned pi05 env.
# python scripts/archive_pi05_uniform_m0.py otherwise puts scripts/ at
# sys.path[0] and fails to locate the sibling agcofl/ package.
if __package__ in (None, ""):
    _project_root = str(Path(__file__).resolve().parents[1])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from agcofl.v140_official_pi05 import IDS, PIN, SEEDS, VARIANTS, preflight, verify, pinned_sha


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(4*1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def collect(source: Path, official_root: Path, assets: Path):
    """Verify every original per-seed summary, demo ID and aggregate tally."""
    aggregated=source/"official_aggregated_summary.json"
    if not aggregated.is_file():
        raise FileNotFoundError(f"No actual 3-pass aggregate at {aggregated}")
    existing=json.loads(aggregated.read_text())
    if (existing.get("type")!="OFFICIAL_PI05_ND_PUSHCUBE_ACTUAL_3PASS"
        or existing.get("method")!=VARIANTS["uniform"]
        or existing.get("official_source_commit")!=PIN
        or existing.get("scope")!="last20"
        or existing.get("task")!="push_cube"
        or existing.get("max_rollout_steps")!=800
        or existing.get("chunk_size")!=20
        or existing.get("full_3pass_complete") is not True
        or existing.get("pooled_trials")!=60):
        raise ValueError("Official aggregate protocol inconsistent; refusing archive")

    if pinned_sha(official_root)!=PIN:
        raise ValueError("Official author source is not at the pinned commit")
    check=preflight("uniform",official_root,assets)
    if not check["ready_for_runtime"]:
        raise ValueError("Current official asset preflight fails: "+", ".join(check["failures"]))

    results=[]
    provenance={"aggregate":{"path":str(aggregated),"sha256":sha256(aggregated)},
                "source_code_commit":PIN, "seed_files":[]}
    rows=[]
    for seed in SEEDS:
        path=source/f"seed_offset_{seed}"/"summary.json"
        config=source/f"config_seed_offset_{seed}.json"
        if not path.is_file() or not config.is_file():
            raise FileNotFoundError(f"Missing official seed summary/config for {seed}")
        raw=json.loads(path.read_text())
        checked=verify(raw,seed)
        cfg=json.loads(config.read_text())
        if (cfg.get("seed_offset_base")!=seed
            or cfg.get("scope")!="last20"
            or cfg.get("tasks")!=["push_cube"]
            or cfg.get("model")!="demo_pi05"
            or cfg.get("max_rollout_steps")!=800):
            raise ValueError(f"seed {seed}: config not frozen official π0.5 protocol")
        results.append(checked)
        provenance["seed_files"].append({
            "seed_offset_base":seed,
            "summary_path":str(path),"summary_sha256":sha256(path),
            "config_path":str(config),"config_sha256":sha256(config)})
        for trial in checked["per_trial"]:
            rows.append({
                "seed_offset_base":seed,
                "demo_id":trial["demo_id"],
                "task":trial["task"],
                "success":trial["success"],
            })

    assert len(rows)==60 and len(results)==3
    success_count=sum(r["success"] for r in rows)
    if success_count!=existing.get("pooled_successes"):
        raise ValueError(f"Aggregate says {existing.get('pooled_successes')} "
                         f"but 60 verified trials say {success_count}")
    by_seed={v["seed_offset_base"]:v for v in results}
    old=existing.get("per_pass")
    if not isinstance(old,list) or len(old)!=3:
        raise ValueError("Expected three aggregate per_pass entries")
    for v in old:
        seed=v.get("seed_offset_base")
        if seed not in by_seed or v!=by_seed[seed]:
            raise ValueError(f"Stored per-pass record inconsistent for seed {seed}")

    output={
        "type":"M0_AUTHOR_PI05_UNIFORM_ND_PUSHCUBE_VERIFIED_60_TRIALS",
        "status":"ACTUAL_OFFICIAL_EVALUATED_ARCHIVE_READY",
        "official_protocol":{
            "task":"push_cube","condition":"without_distraction/without_distraction",
            "scope":"last20","model":"demo_pi05",
            "checkpoint_variant":"vid_cond_pi05_uniform",
            "max_rollout_steps":800,"action_chunk_size":20,
            "demo_ids":list(IDS),"seed_offsets":list(SEEDS)},
        "source_revision":PIN,"pooled_successes":success_count,
        "pooled_trials":60,"success_rate":success_count/60,
        "per_seed":[{"seed_offset_base":v["seed_offset_base"],
                     "successes":v["success"],"trials":v["total"],
                     "success_rate":v["success_rate"]} for v in results],
        "per_trial":rows,"provenance":provenance,
        "pi05_base_weight_source":"robotgeneralist/openpi_checkpoint_mirrors2 pinned HF third-party mirror (GCS source identity not independently proven)",
        "not_a_cross_behavior_capability_result":True,
        "no_simulation_or_training_in_archive_step":True,
    }
    return output


def _write_no_overwrite(path:Path, data:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:
            raise FileExistsError(f"Existing archive differs; investigate, do NOT overwrite: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:
        tmp=Path(handle.name)
        handle.write(data)
    try:
        # Another process could have created it; fail rather than clobber.
        if path.exists():
            raise FileExistsError(path)
        tmp.rename(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def archive(report:dict, out:Path):
    _write_no_overwrite(out/"verified_summary.json",
                       (json.dumps(report,indent=2,ensure_ascii=False)+"\n").encode())
    import io
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=(
        "seed_offset_base","demo_id","task","success"))
    writer.writeheader()
    for row in report["per_trial"]:
        writer.writerow(row)
    _write_no_overwrite(out/"per_trial_60.csv",stream.getvalue().encode())
    breakdown="; ".join(
        f"seed {v['seed_offset_base']}: {v['successes']}/{v['trials']}"
        for v in report["per_seed"])
    md=(f"# Official RoboReel π0.5 Uniform — M0 frozen results\n\n"
        f"**Actual physical evaluation:** {report['pooled_successes']}/"
        f"{report['pooled_trials']} = {report['success_rate']*100:.2f}%\n\n"
        f"**Condition:** PushCube, no-distraction human / no-distraction env, "
        f"20 held-out demo IDs 0080–0099 repeated for 3 seed offsets.\n\n"
        f"**Per-seed:** {breakdown}\n\n"
        f"**Author source pinned:** `{PIN}`. Original π0.5 Uniform checkpoint "
        f"and original SigLIP processing; base weights from a third-party HF mirror "
        f"with GCS-byte identity not independently confirmed.\n\n"
        f"Raw compact results: `verified_summary.json`, `per_trial_60.csv`. "
        f"Both are derived only from real saved official summary files and "
        f"validated against their SHA-256 hashes; no simulation was run by "
        f"the archiving script.\n\n"
        f"This is an **official-benchmark baseline** result, not a successful "
        f"same-goal Pick→Push capability-adaptation experiment.\n")
    _write_no_overwrite(out/"SUMMARY.md",md.encode())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/v140_pi05_uniform"))
    parser.add_argument("--official-root",type=Path,default=Path("third_party/roboreel_models"))
    parser.add_argument("--assets",type=Path,default=Path("outputs/v120_baselines"))
    parser.add_argument("--out",type=Path,default=Path("research_reports/2026-11-11_m0_pi05_uniform"))
    args=parser.parse_args(argv)
    result=collect(args.source.resolve(),args.official_root.resolve(),args.assets.resolve())
    archive(result,args.out)
    print("M0_PI05_UNIFORM_ARCHIVED",json.dumps({
        "pooled_successes":result["pooled_successes"],
        "pooled_trials":result["pooled_trials"],
        "per_seed":result["per_seed"],
        "out":str(args.out),"official_source_revision":result["source_revision"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
