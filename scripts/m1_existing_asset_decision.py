#!/usr/bin/env python3
"""TRO1 M1: read-only evidence ledger for existing B2, B3 and official assets.

CPU + Python stdlib only. No sim import, environment reset, training, network,
new grasp attempts, relabelled source demonstrations, or original-data writes.
Do not confuse a successful source robot rollout with a validated human video.

Uses already-frozen same-goal B2 validator; records on-disk H5/video existence
independently from actual PhysX success, and does not infer causal necessity
from a failed direct-push script or an unexecuted strong TAMP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.m1b2_validate_paired_physics import summarize as verify_b2
from scripts.m1a_inventory_real_assets import audit as prior_official_inventory

B2_ROOT = Path('outputs/m1b2_same_goal_dev')
B3_SOURCE = Path('outputs/m1_source_geom_aligned_topdown_v1/source_aligned_topdown')
B3_TARGET = Path('outputs/m1_original_scene_cartesian_pusher_v6/target_cartesian_contact_fixed')
B3_PLAN = Path('outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_audit.json')
OUTPUT = Path('outputs/m1_existing_asset_decision/decision.json')
MARKDOWN = Path('outputs/m1_existing_asset_decision/decision.md')


def _rel(path: Path, project: Path) -> str:
    try:
        return str(path.relative_to(project))
    except ValueError:
        return str(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _positive_int(value):
    return type(value) is int and value > 0


def _vector3(v):
    return (isinstance(v, list) and len(v) == 3 and all(
        type(x) in (int, float) and math.isfinite(x) for x in v))


def _same(a, b, tol=1e-4):
    return _vector3(a) and _vector3(b) and math.dist(a, b) <= tol


def case_evidence(root: Path, project: Path, expected_actions: int | None = None):
    """Physical metadata, raw trace integrity and MP4/H5 PRESENCE are separate.

    MP4 present != visual contents reviewed; H5 present != successful grasp.
    Large H5 and videos are never read or hashed; no needless IO on RTX4090.
    """
    status = root/'status.json'
    trace = root/'physical_trace.json'
    result = {
        'root': _rel(root, project),
        'status_exists': status.is_file(),
        'status_sha256': _sha(status) if status.is_file() else None,
        'trace_exists': trace.is_file(),
        'trace_sha256': _sha(trace) if trace.is_file() else None,
        'trace_actions': None,
        'trace_length_matches_status': None,
    }
    if trace.is_file():
        rows = _json(trace)
        if not isinstance(rows, list):
            raise ValueError(f'Physical trace not a list: {trace}')
        result['trace_actions'] = len(rows)
        if expected_actions is not None:
            result['trace_length_matches_status'] = len(rows) == expected_actions
        # An available but contradictory trace is a hard evidence failure.
        if expected_actions is not None and len(rows) != expected_actions:
            raise ValueError(f'Trace/status action mismatch: {trace}: {len(rows)} != {expected_actions}')
    # Do not follow symlinks outside the scene asset folder.
    video_files = sorted(p for p in root.rglob('*.mp4') if p.is_file() and not p.is_symlink() and p.stat().st_size > 512)
    h5_files = sorted(p for p in root.rglob('*.h5') if p.is_file() and not p.is_symlink() and p.stat().st_size > 512)
    result['videos'] = [_rel(p, project) for p in video_files[:20]]
    result['h5_files'] = [_rel(p, project) for p in h5_files[:20]]
    result['video_file_exists_not_content_validated'] = bool(video_files)
    result['h5_file_exists_not_content_validated'] = bool(h5_files)
    result['has_full_local_artifact_triple'] = (
        result['trace_exists'] and bool(video_files) and bool(h5_files)
        and result['trace_length_matches_status'] is True)
    return result


def b2_review(project: Path):
    root = project/B2_ROOT
    # Existing B2 validator checks same initial state/goal, seed, real action
    # metadata, and named push/clear/source gates. It creates no new files.
    gate = verify_b2(root)
    out = {
        'scene': 'M1-B2', 'source_goal_x_m': .150,
        'gate_status': gate['status'],
        'physics_pair_from_existing_statuses': False,
        'causal_insert_result': 'NOT_ESTABLISHED',
        'source_demonstration_kind': 'SIMULATED_ROBOT_MOTIONPLANNING_NOT_HUMAN_DEMO',
        'source_video_human_provenance': False,
        'video_information_gain_tested': False,
        'equal_information_strong_tamp_executed': False,
        'learned_method_executed': False,
        'cases': {},
    }
    if gate.get('status') in ('INCOMPLETE_MISSING_ACTUAL_ROLLOUTS', 'CASE_EXECUTION_ERROR_NO_PAIRED_PHYSICS_CLAIM'):
        out['reason'] = 'Existing B2 statuses missing or original execution error'
        return out
    for mode in ('source', 'target_direct', 'target_clear'):
        state = _json(root/mode/'status.json')
        count = state.get('steps', state.get('actual_rollout_steps'))
        evidence = case_evidence(root/mode, project, count)
        out['cases'][mode] = {
            'real_physics_success': state['physically_verified_matched_success'],
            'steps': count,
            'error': state.get('error'),
            'artifacts': evidence,
        }
    successful_pair = (gate.get('matched_physics_start_and_goal') is True
                       and gate.get('source_pick_place_valid') is True
                       and (gate.get('target_direct_push_success') is True
                            or gate.get('target_clear_then_push_success') is True))
    out['physics_pair_from_existing_statuses'] = successful_pair
    out['raw_traces_complete'] = all(
        item['artifacts']['trace_length_matches_status'] is True
        for item in out['cases'].values())
    out['source_robot_video_on_disk'] = out['cases']['source']['artifacts']['video_file_exists_not_content_validated']
    out['source_robot_h5_on_disk'] = out['cases']['source']['artifacts']['h5_file_exists_not_content_validated']
    if successful_pair and gate['target_direct_push_success'] is True:
        out['causal_insert_result'] = 'REFUTED_BY_SUCCESSFUL_DIRECT_PUSH'
        out['decision'] = 'KEEP_AS_TRIVIAL_SANITY_CONTROL_NOT_INSERT_EVIDENCE'
    elif (successful_pair and gate['target_direct_push_success'] is False
          and gate['target_clear_then_push_success'] is True
          and gate['target_only_blocker_displacement_observed'] is True):
        out['causal_insert_result'] = 'SCRIPTED_CONTRAST_ONLY_NEEDS_STRONG_TAMP_AND_CAUSAL_PROOF'
        out['decision'] = 'CANDIDATE_FOR_ADDITIONAL_NONPHYSICS_CHECK_NOT_METHOD_GO'
    else:
        out['decision'] = 'NOT_DISCRIMINATIVE_OR_NOT_PAIRED'
    out['gate_flags'] = {
        'source_success': gate['source_pick_place_valid'],
        'direct_push_success': gate['target_direct_push_success'],
        'clear_push_success': gate['target_clear_then_push_success'],
        'blocker_moved_in_clear': gate['target_only_blocker_displacement_observed'],
        'pilot_contrast_pass': gate['pilot_contrast_pass'],
    }
    return out


def b3_review(project: Path):
    source = project/B3_SOURCE
    target = project/B3_TARGET
    source_state = _json(source/'status.json')
    target_state = _json(target/'status.json')
    plan = _json(project/B3_PLAN)
    out = {
        'scene': 'M1-B3', 'goal_x_m': .210,
        'physics_pair_from_existing_statuses': False,
        'target_success': False,
        'source_success': False,
        'source_grasp_executor_decision': 'ENGINEERING_NO_GO_FROZEN',
        'strong_tamp_executed': False,
        'video_information_gain_tested': False,
        'cases': {},
        'planning_audit': 'REPORT_NOT_FOUND_ON_DISK',
    }
    if source_state is not None:
        steps = source_state.get('actual_steps')
        source_ok = (source_state.get('version') == 'm1_source_geom_aligned_topdown_v1'
                     and source_state.get('scene') == 'AGCOFLM1B3Corridor-v0'
                     and source_state.get('seed') == 120
                     and source_state.get('success') is False
                     and steps == 65)
        if not source_ok:
            raise ValueError('Frozen B3 source metadata changed or does not match real 65-step failure')
        out['cases']['source_failed'] = case_evidence(source, project, steps)
        out['source_real_failed_episode_preserved'] = True
    else:
        out['source_real_failed_episode_preserved'] = False
    if target_state is not None:
        steps = target_state.get('steps')
        target_ok = (target_state.get('scene') == 'AGCOFLM1B3Corridor-v0'
                     and target_state.get('seed') == 120
                     and target_state.get('success') is True
                     and steps == 66
                     and target_state.get('clearance_valid') is True
                     and target_state.get('push_only', {}).get('valid') is True
                     and _vector3(target_state.get('goal'))
                     and abs(target_state['goal'][0] - .210) < .001
                     and target_state.get('goal_error_m', 1.) < .020)
        if not target_ok:
            raise ValueError('B3 target v6 saved status violates frozen 66-step physical success contract')
        out['target_success'] = True
        out['cases']['target_v6'] = case_evidence(target, project, steps)
    if plan is not None:
        pl = plan.get('planning') or {}
        if (plan.get('decision') != 'INCONCLUSIVE_IK_OR_COLLISION_FAILURE'
                or pl.get('ik_solutions_returned') != 0
                or plan.get('new_physx_actions') != 0
                or plan.get('authorizes_physics') is not False):
            raise ValueError('B3 planning report differs from confirmed frozen IK-zero decision')
        out['planning_audit'] = 'IK_0_SCREW_FAILED_NO_PHYSICS_AUTHORIZATION'
        out['planning_audit_file_sha256'] = _sha(project/B3_PLAN)
    out['decision'] = ('TARGET_ONLY_SUCCESS_SOURCE_PAIR_BLOCKED'
                       if out['target_success'] else 'TARGET_EVIDENCE_MISSING')
    return out


def audit(project: Path):
    root = project.resolve()
    b2 = b2_review(root)
    b3 = b3_review(root)
    official = prior_official_inventory(root)
    official_stats = {
        'm0_pi05_verified_report_on_disk': official['m0_official_pi05_archived_and_verified'],
        'official_train_pick_human_video_files': len(official['local_train_pick_human_front_mp4']),
        'official_train_push_human_video_files': len(official['local_train_push_human_front_mp4']),
        'official_train_push_robot_h5_camera_triplets': len(official['local_train_push_robot_demos_with_h5_and_two_cameras']),
        'original_pickcube_and_pushcube_not_same_goal': True,
        'official_assets_do_not_by_themselves_form_custom_same_goal_pairs': True,
    }
    # A single successful path never proves all alternatives impossible.
    # Even a simulated-source robot video with identical goal might not
    # contribute task information compared with goal+current RGB observations.
    prospective_insert = (
        b2.get('causal_insert_result') == 'SCRIPTED_CONTRAST_ONLY_NEEDS_STRONG_TAMP_AND_CAUSAL_PROOF'
        and b2.get('source_robot_video_on_disk', False)
        and b2.get('raw_traces_complete', False))
    decision = ('CANDIDATE_REQUIRES_VIDEO_INFORMATION_AND_STRONG_TAMP_GATE'
                if prospective_insert else 'NO_GO_FOR_CURRENT_M1_INSERT_METHOD_START')
    return {
        'type': 'TRO1_M1_EXISTING_PHYSICAL_ASSETS_DECISION_V1',
        'decision': decision,
        'new_physx_actions': 0,
        'new_training_runs': 0,
        'new_downloads': 0,
        'source_goal_equality_checked_by_existing_b2_validator': True,
        'source_video_present_is_not_human_demo_or_useful_video_proof': True,
        'no_method_or_paper_novelty_claimed': True,
        'b2': b2, 'b3': b3, 'official': official_stats,
        'next_action': (
            'KEEP_B2_AS_SANITY_AND_B3_AS_ENGINEERING_NEGATIVE; '
            'SEARCH_INDEPENDENTLY_VALIDATED_MATCHED_GOAL_SOURCE_TARGET_ASSETS; '
            'DO_NOT_BUILD_M1B5_OR_TRAIN_M2_UNTIL_VIDEO_INFORMATION_AND_TAMP_GAP_PASS'
        ),
    }


def markdown(report):
    b2, b3, official = report['b2'], report['b3'], report['official']
    lines = [
        '# TRO1 M1 existing-assets gate — real data only', '',
        f"**Decision:** `{report['decision']}`", '',
        'This is a read-only local evidence inventory, not a learned-method result.', '',
        '| Scene | Successful actual source | Successful actual target | Direct push | Causal insert necessary | Verdict |',
        '|---|---|---|---|---|---|',
    ]
    flags = b2.get('gate_flags', {})
    src = 'YES' if flags.get('source_success') else 'NO/UNKNOWN'
    target = 'YES' if flags.get('direct_push_success') or flags.get('clear_push_success') else 'NO/UNKNOWN'
    direct = 'SUCCESS' if flags.get('direct_push_success') else 'NO/UNKNOWN'
    lines.append(f"| B2 x=.150 | {src} | {target} | {direct} | NOT DEMONSTRATED | {b2.get('decision','MISSING')} |")
    lines.append(f"| B3 x=.210 | NO | {'YES (v6)' if b3['target_success'] else 'UNKNOWN'} | NOT PROVEN | NOT DEMONSTRATED | {b3['decision']} |")
    lines.extend([
        '', '## Evidence boundary', '',
        f"- B2 physics matched pair from statuses: `{b2['physics_pair_from_existing_statuses']}`; direct push success: `{flags.get('direct_push_success')}`.",
        f"- B2 source simulated ROBOT video file on disk: `{b2.get('source_robot_video_on_disk',False)}`; H5: `{b2.get('source_robot_h5_on_disk',False)}`. Neither implies a HUMAN demonstrator or useful video information.",
        f"- B3 source 65-step failure preserved: `{b3['source_real_failed_episode_preserved']}`; target v6 66-step success: `{b3['target_success']}`; planning audit: `{b3['planning_audit']}`.",
        f"- Official train pick human video files: `{official['official_train_pick_human_video_files']}`; robot Push H5+two-camera triples: `{official['official_train_push_robot_h5_camera_triplets']}`. Original PickCube/PushCube goals are DIFFERENT.",
        '- Goal-only vs video-dependent task-information gain: NOT_RUN.',
        '- Equally informed capability-aware TAMP: NOT_RUN.',
        '- Real grasp source under B3 and strong new learning method M2: NOT_RUN / NOT ESTABLISHED.',
        '', '## Next scientific decision', '',
        'Do not run M1-B5, retune the frozen Panda grasp, or train a causal editor. '
        'Preserve B2 as a working non-discriminative sanity scene and B3 as a blocked-source engineering negative. '
        'A new method requires a genuinely matched-goal source/target demonstration, '
        'an identifiable fact carried by the source video beyond the target goal/observations, '
        'and a specified failure mode of a strong equally informed planner.',
        '', 'The JSON next to this report includes file-presence and SHA evidence for small JSON files. '
        'Presence of MP4/H5 is NOT proof of their content; large binary files were not inspected or copied.',
        '',
    ])
    return '\n'.join(lines)


def write_once(path: Path, text: str):
    if path.is_file():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError('Refusing to overwrite previous evidence report: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, default=OUTPUT)
    p.add_argument('--md', type=Path, default=MARKDOWN)
    a = p.parse_args(argv)
    result = audit(a.project)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n'
    preview = markdown(result)
    # Independent, append-never output folder; never change original H5/JSON.
    write_once(a.out, payload)
    write_once(a.md, preview)
    print('M1_EXISTING_ASSET_DECISION',json.dumps({
        'decision': result['decision'],
        'b2_pair': result['b2']['physics_pair_from_existing_statuses'],
        'b2_direct_push_success': result['b2'].get('gate_flags', {}).get('direct_push_success'),
        'b2_source_video_file_on_disk': result['b2'].get('source_robot_video_on_disk'),
        'b3_source_success': result['b3']['source_success'],
        'b3_target_success': result['b3']['target_success'],
        'm1_insert_scientifically_validated': False,
        'new_physx_actions': 0, 'new_training_runs': 0,
        'report_json': str(a.out), 'report_md': str(a.md),
    },ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
