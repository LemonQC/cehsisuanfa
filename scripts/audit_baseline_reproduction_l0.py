#!/usr/bin/env python3
"""CPU-only, read-only local checkout + official-results audit of TRO1 B01..B14.

Does not fetch repos, train, infer, run PhysX, test GPU readiness, or assert
any official cross-method reproducibility merely because a file exists.
Publishing uses the existing dedicated evidence branch, not main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = Path("outputs/research_runs/baseline_readiness_l0")

# This is a *local checkout* check only, NOT an author-release audit.
# Every row must have an explicit expected project-local entrypoint or none.
ROWS = (
    ("B01", "RobotOnlyTransformerBC", "REIMPL", ("scripts/prepare_b01_robot_train.py",)),
    ("B02", "RobotOnlyDiffusionPolicy", "REIMPL", ("scripts/train_robot_only_diffusion_policy.py",)),
    ("B03", "RoboReelVideoPi05Uniform", "AUTHOR", ("third_party/roboreel_models/scripts/train_vid_cond_pi05.py", "third_party/roboreel_models/scripts/evaluation.py")),
    ("B04", "RoboReelVideoPi05Keyframes", "AUTHOR", ("third_party/roboreel_models/scripts/extract_keyframes.py", "third_party/roboreel_models/scripts/evaluation.py")),
    ("B05", "RoboReelVid2Robot", "AUTHOR", ("third_party/roboreel_models/scripts/train_vid2robot.py", "third_party/roboreel_models/scripts/evaluation.py")),
    ("B06", "RoboReelUniSkillPolicy", "AUTHOR", ("third_party/roboreel_models/scripts/train_uniskill_policy.py", "third_party/roboreel_models/scripts/evaluation.py")),
    ("B07", "RoboReelVlmPi05LocalQwen", "AUTHOR", ("third_party/roboreel_models/scripts/annotate_human_videos.py", "third_party/roboreel_models/scripts/train_vlm_pi05.py")),
    ("B08", "SeeDo", "AUTHOR_NOT_IN_ROBOREEL_RELEASE", ()),
    ("B09", "PointPolicy", "INDEPENDENT_AUTHOR", ("third_party/Point-Policy/point_policy",)),
    ("B10", "CrossBehaviorObjectFlow2026", "AUTHOR_SOURCE_UNVERIFIED", ()),
    ("B11", "Im2Flow2Act", "INDEPENDENT_AUTHOR", ("third_party/im2Flow2Act/scripts",)),
    ("B12", "ECFlow", "INDEPENDENT_AUTHOR", ("third_party/EC-Flow/inference.py",)),
    ("B13", "DINOv2TemporalPilot", "DIAGNOSTIC_NOT_FULL_POLICY", ("agcofl/vit_flow_pilot.py",)),
    ("B14", "GoalWaypointTAMPOracle", "HETEROGENEOUS_CONTROLS", ("agcofl/m1_original_scene_cartesian_pusher_v6.py",)),
)


def _existing_regular(project: Path, relative: str) -> bool:
    """Reject symlinked entries/ancestors, do not follow into private or external paths."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    cur = project
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            return False
    return cur.exists() and (cur.is_file() or cur.is_dir())


def _safe_json(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 256_000:
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, UnicodeError, OSError):
        return None


def official_evidence(project: Path) -> dict:
    """Only two already-published PushCube ND official anchors can pass."""
    ret = {"B03": None, "B05": None}
    b03 = _safe_json(project / "research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json")
    if (b03 and b03.get("status") == "ACTUAL_OFFICIAL_EVALUATED_ARCHIVE_READY" and
        b03.get("pooled_successes") == 46 and b03.get("pooled_trials") == 60 and
        (b03.get("official_protocol") or {}).get("task") == "push_cube"):
        ret["B03"] = {"task": "push_cube", "condition": "nd", "successes": 46, "trials": 60,
                      "artifact": "research_reports/2026-11-11_m0_pi05_uniform/verified_summary.json"}
    b05 = _safe_json(project / "research_reports/2026-11-09_v130/official_aggregated_summary.json")
    if (b05 and b05.get("full_3pass_complete") is True and
        b05.get("pooled_successes") == 3 and b05.get("pooled_trials") == 60 and
        b05.get("task") == "push_cube"):
        ret["B05"] = {"task": "push_cube", "condition": "nd", "successes": 3, "trials": 60,
                      "artifact": "research_reports/2026-11-09_v130/official_aggregated_summary.json"}
    return ret


def inventory(project: Path) -> dict:
    project = project.expanduser().resolve()
    observed = official_evidence(project)
    rows = []
    for code, name, classification, paths in ROWS:
        checks = [{"relative_path": p, "exists_in_local_checkout": _existing_regular(project, p)}
                  for p in paths]
        available = bool(checks) and all(x["exists_in_local_checkout"] for x in checks)
        rows.append({
            "id": code, "name": name, "reproduction_kind": classification,
            "local_entrypoints": checks,
            "all_expected_local_entrypoints_present": available,
            "source_revision_verified": False,
            "weights_sha_verified_in_this_audit": False,
            "native_forward_executed_by_this_audit": False,
            "new_physx_rollouts": 0,
            "new_training_runs": 0,
            "locked_existing_official_pushcube_nd_evidence": observed.get(code),
            "next_gate": ("AUTHOR_RELEASE_OR_ALTERNATIVE_CODE_AUDIT" if not paths else
                          "VERIFY_VERSION_AND_DEPENDENCIES" if available else
                          "ACQUIRE_OR_BUILD_EXACT_ENTRYPOINT"),
        })
    assert len(rows) == 14 and len({r["id"] for r in rows}) == 14
    return {
        "schema": "TRO1_BASELINE_LOCAL_READINESS_L0_V1",
        "scope": "LOCAL_ENTRYPOINT_PRESENCE_ONLY_NOT_NATIVE_REPRODUCTION",
        "methods_count": 14,
        "existing_official_repro_method_ids": sorted(x for x in observed if observed[x]),
        "source_revisions_not_verified": True,
        "training_or_eval_was_not_executed": True,
        "gated_datasets_checkpoint_downloads": 0,
        "physx_steps": 0,
        "methods": rows,
    }


def write_once(project: Path, value: dict) -> tuple[Path, bytes]:
    raw = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    uid = hashlib.sha256(raw).hexdigest()[:16]
    path = project / OUTPUT / ("inventory_" + uid + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise FileExistsError("L0_REPORT_PATH_CHANGED_REFUSE_OVERWRITE")
    else:
        with path.open("xb") as out:
            out.write(raw)
    return path, raw


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=PROJECT)
    parser.add_argument("--publish",action="store_true",help="Opt-in upload sanitized manifest to dedicated evidence branch")
    args=parser.parse_args(argv)
    project=args.project.expanduser().resolve()
    try:
        report=inventory(project)
        file,data=write_once(project,report)
        published=None
        if args.publish:
            from scripts.publish_research_evidence import assert_evidence_branch_exists, write_immutable_report
            assert_evidence_branch_exists()
            uid=hashlib.sha256(data).hexdigest()[:16]
            path=f"evidence/baseline_l0/baseline_source_inventory_{uid}/inventory.json"
            result=write_immutable_report(path,data)
            published={"result":result,"url":f"https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/{path}"}
        print("TRO1_BASELINE_L0_READINESS",json.dumps({
            "status":"LOCAL_ENTRYPOINT_AUDIT_ONLY_NOT_REPRODUCED",
            "rows":14,
            "official_repro_anchor_ids":report["existing_official_repro_method_ids"],
            "local_entrypoints_found":sum(x["all_expected_local_entrypoints_present"] for x in report["methods"]),
            "artifact_relative":file.relative_to(project).as_posix(),
            "published":published,
            "new_physx_actions":0,"new_training_runs":0,
        },ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        # Do not publish local project paths, exception text, keys or token values.
        print("TRO1_BASELINE_L0_BLOCKED",json.dumps({"error_class":type(exc).__name__,
              "no_network_downloads":True,"new_physx_actions":0}),flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
