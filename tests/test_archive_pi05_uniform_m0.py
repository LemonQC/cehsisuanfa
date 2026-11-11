"""CPU-only M0 archive tests: 60 real-format records, no simulation."""
import csv
import json
from pathlib import Path

import pytest

from agcofl.v140_official_pi05 import CELL, IDS, PIN, SEEDS, VARIANTS, verify
from scripts import archive_pi05_uniform_m0 as a


def _setup(tmp_path, successes_by_seed=(15,15,16)):
    root=tmp_path/"official"
    root.mkdir()
    assets=tmp_path/"assets"
    source=tmp_path/"run"
    per=[]
    for seed,n in zip(SEEDS,successes_by_seed):
        rows=[{"task":"push_cube","demo_id":i,"success":i<80+n} for i in IDS]
        raw={
            "model":"demo_pi05","runner":"demo_pi05","scope":"last20",
            "tasks":["push_cube"],"chunk_size":20,"max_rollout_steps":800,
            "seed_offset_base":seed,"demo_ids_no_distract":IDS,
            "human_demo_and_env_types":[["without_distraction","without_distraction"]],
            "per_trial":{CELL:rows},
            "table":{CELL:{"success":n,"total":20,"rate":n/20}},
        }
        path=source/f"seed_offset_{seed}"/"summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(raw))
        cfg={"seed_offset_base":seed,"scope":"last20",
             "tasks":["push_cube"],"model":"demo_pi05",
             "max_rollout_steps":800}
        (source/f"config_seed_offset_{seed}.json").write_text(json.dumps(cfg))
        per.append(verify(raw,seed))
    aggregate={
        "type":"OFFICIAL_PI05_ND_PUSHCUBE_ACTUAL_3PASS",
        "method":VARIANTS["uniform"],
        "official_source_commit":PIN,"scope":"last20",
        "task":"push_cube","max_rollout_steps":800,"chunk_size":20,
        "full_3pass_complete":True,"pooled_trials":60,
        "pooled_successes":sum(successes_by_seed),
        "per_pass":per,
    }
    (source/"official_aggregated_summary.json").write_text(json.dumps(aggregate))
    return source,root,assets


def test_aggregate_archive_extracts_real_60_trial_shapes(tmp_path,monkeypatch):
    source,root,assets=_setup(tmp_path)
    monkeypatch.setattr(a,"pinned_sha",lambda *_:PIN)
    monkeypatch.setattr(a,"preflight",lambda *_:{"ready_for_runtime":True})
    report=a.collect(source,root,assets)
    assert report["pooled_successes"]==46
    assert report["pooled_trials"]==60
    assert len(report["per_trial"])==60
    assert {v["seed_offset_base"] for v in report["per_trial"]}==set(SEEDS)
    assert sum(r["success"] for r in report["per_trial"])==46
    assert len(report["provenance"]["seed_files"])==3
    out=tmp_path/"frozen"
    a.archive(report,out)
    with (out/"per_trial_60.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==60
    assert "76.67%" in (out/"SUMMARY.md").read_text()
    # Immutable archive: repeat with identical inputs is safe.
    a.archive(report,out)
    (out/"SUMMARY.md").write_text("TAMPERED")
    with pytest.raises(FileExistsError):
        a.archive(report,out)


def test_archive_refuses_inconsistent_aggregate(tmp_path,monkeypatch):
    source,root,assets=_setup(tmp_path)
    monkeypatch.setattr(a,"pinned_sha",lambda *_:PIN)
    monkeypatch.setattr(a,"preflight",lambda *_:{"ready_for_runtime":True})
    path=source/"official_aggregated_summary.json"
    data=json.loads(path.read_text())
    data["pooled_successes"]=60
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="60 verified trials"):
        a.collect(source,root,assets)


def test_archive_refuses_missing_seed_or_wrong_source(tmp_path,monkeypatch):
    source,root,assets=_setup(tmp_path)
    monkeypatch.setattr(a,"pinned_sha",lambda *_:"WRONG")
    monkeypatch.setattr(a,"preflight",lambda *_:{"ready_for_runtime":True})
    with pytest.raises(ValueError,match="pinned commit"):
        a.collect(source,root,assets)
    monkeypatch.setattr(a,"pinned_sha",lambda *_:PIN)
    (source/"seed_offset_100000"/"summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        a.collect(source,root,assets)
