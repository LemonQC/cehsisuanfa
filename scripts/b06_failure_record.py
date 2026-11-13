#!/usr/bin/env python3
"""Report a B06 setup/runtime failure without publishing credentials or raw logs.

May run in an arbitrary existing Python environment. Never imports torch.
The report can be retried with the same arguments after a GitHub outage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
STAGES = {
    'project_gate', 'bounded_video_subset', 'author_source_submodules',
    'project_local_conda_env', 'install_uniskill_deps',
    'pin_hf_revisions_and_disk_gate', 'download_pinned_official_uniskill_policy',
    'download_pinned_author_idm', 'preflight_before_skill_extraction',
    'extract_two_genuine_human_skills', 'final_preflight',
    'republish_completed_without_physx', 'real_physx_bounded_dev_and_publish',
}


def make_record(stage: str, exit_code: int) -> dict:
    if stage not in STAGES or not isinstance(exit_code, int) or not 1 <= exit_code <= 255:
        raise ValueError('INVALID_B06_STAGE_OR_EXIT_CODE')
    status = ('PARTIAL_PHYSX_POSSIBLE_DO_NOT_RERUN'
              if stage == 'real_physx_bounded_dev_and_publish' else 'SETUP_BLOCKED_NO_PHYSX_STARTED_IN_THIS_ATTEMPT')
    return {
        'schema': 'TRO1_B06_UNISKILL_BLOCKER_V1',
        'status': status,
        'baseline': 'B06',
        'stage': stage,
        'exit_code': exit_code,
        'gpu_server_rollouts_verified_by_this_report': False,
        'no_secrets_or_raw_exception_text_in_report': True,
        'manual_rerun_of_physx_forbidden_if_partial_results_exist': True,
    }


def save_local(project: Path, record: dict) -> tuple[Path, bytes, str]:
    raw = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    root = project / 'outputs/research_runs/b06_blocked'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'b06_blocked_{digest}.json'
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise FileExistsError('B06_BLOCKER_IMMUTABILITY_CONFLICT')
    else:
        with path.open('xb') as f: f.write(raw)
    return path, raw, digest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=PROJECT)
    p.add_argument('--stage', required=True, choices=sorted(STAGES))
    p.add_argument('--exit-code', required=True, type=int)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args(argv)
    record = make_record(args.stage, args.exit_code)
    path, raw, digest = save_local(args.project.resolve(), record)
    print('B06_BLOCKED_SNAPSHOT', json.dumps({'stage':args.stage, 'exit_code':args.exit_code,
         'relative_file':str(path.relative_to(args.project.resolve()))}), flush=True)
    if args.publish:
        try:
            from scripts.publish_research_evidence import assert_evidence_branch_exists, write_immutable_report
            assert_evidence_branch_exists()
            remote=f'evidence/runs/b06_blocked_{digest}/evidence.json'
            val=write_immutable_report(remote,raw)
            print('B06_BLOCKED_PUBLISHED',json.dumps({'status':val,
                 'url':'https://github.com/LemonQC/cehsisuanfa/blob/tro1-experiment-evidence/'+remote}),flush=True)
        except Exception:
            # Network failure must never erase the original B06 failure.
            print('B06_BLOCKED_LOCAL_ONLY_PUBLISH_FAILED', flush=True)
            return 5
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
