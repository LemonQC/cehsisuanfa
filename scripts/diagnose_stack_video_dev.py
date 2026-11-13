#!/usr/bin/env python3
"""Read-only forensic analysis of frozen RoboReel StackCups TRAIN-video DEV v1.

Reads the four real rollout action arrays, trial status and optional MP4s.
Does not import the simulator, model, JAX, OpenPI, or access a network.
Writes only into a NEW separate diagnosis directory; original evidence stays intact.
All metrics describe differences and cannot establish useful conditioning or T-RO novelty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_TASK = 'stack_cups'
EXPECTED_IDS = (0, 1)
PREFIXES = (1, 20, 50, 100, 200, 400, 800)
STORY_STEPS = (0, 20, 60, 120, 200, 300, 400, 500, 650, 799)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_source(root: Path):
    """Fail closed on absent, incomplete, or changed real evidence."""
    root = root.resolve()
    results = root / 'results.json'
    if not results.is_file():
        raise FileNotFoundError('Original frozen results.json absent: ' + str(results))
    doc = json.loads(results.read_text(encoding='utf-8'))
    if doc.get('status') != 'DEV_COMPLETE_NOT_FORMAL' or doc.get('task') != EXPECTED_TASK:
        raise ValueError('NOT_THE_FROZEN_OFFICIAL_STACK_CUPS_DEV_RESULT')
    trials = doc.get('trials')
    if (doc.get('paired_episodes') != 2 or doc.get('real_rollouts') != 4
            or not isinstance(trials, list) or len(trials) != 4):
        raise ValueError('EXPECTED_EXACTLY_FOUR_REAL_STACK_CUPS_TRIALS')
    by_episode = {}
    actions = {}
    for trial in trials:
        ep = trial.get('episode')
        source = trial.get('source_train_demo_id')
        if type(ep) is not int or ep not in (0, 1) or type(source) is not int or source not in EXPECTED_IDS:
            raise ValueError('UNEXPECTED_EPISODE_OR_TRAIN_SOURCE')
        key = (ep, source)
        if key in actions:
            raise ValueError('DUPLICATE_TRIAL')
        expected_path = root / f'episode_{ep:02d}' / f'source_{source:04d}' / 'actions.npy'
        expected_path = expected_path.resolve()
        if not expected_path.is_relative_to(root):
            raise ValueError('INVALID_SOURCE_PATH')
        if str(expected_path) != str(Path(trial.get('actions_path', '')).resolve()):
            raise ValueError('TRIAL_ACTION_PATH_MISMATCH')
        stored_hash = trial.get('actions_sha256')
        if not stored_hash or sha256(expected_path) != stored_hash:
            raise ValueError('ACTION_HASH_MISMATCH: ' + str(expected_path))
        # Numpy .npy only, never allow_pickle.
        a = np.load(expected_path, allow_pickle=False)
        if (a.ndim != 2 or a.shape[1] != 8 or a.shape[0] != trial.get('steps')
                or a.shape[0] < 1 or not np.isfinite(a).all()):
            raise ValueError('INVALID_RECORDED_ACTION_SHAPE_OR_STEPS')
        actions[key] = a.astype(np.float64, copy=False)
        by_episode.setdefault(ep, []).append(trial)
    for ep, items in by_episode.items():
        if sorted(item['source_train_demo_id'] for item in items) != [0, 1]:
            raise ValueError('PAIR_MISSING_A_REAL_TRAIN_VIDEO')
        if (len(set(item['reset_seed'] for item in items)) != 1
                or len(set(item['policy_rng_seed'] for item in items)) != 1):
            raise ValueError('PAIR_SEED_OR_RNG_NOT_MATCHED')
    if len(by_episode) != 2:
        raise ValueError('TWO_PAIRED_SCENES_REQUIRED')
    return doc, actions, by_episode


def range_metrics(a: np.ndarray, b: np.ndarray, limit: int) -> dict:
    n = min(len(a), len(b), limit)
    d = a[:n] - b[:n]
    norms = np.linalg.norm(d, axis=1)
    ref = (np.linalg.norm(a[:n], axis=1) + np.linalg.norm(b[:n], axis=1)) / 2
    return {
        'steps_compared': int(n),
        'mean_8d_action_l2': float(norms.mean()),
        'max_8d_action_l2': float(norms.max()),
        'mean_7d_arm_l2': float(np.linalg.norm(d[:, :7], axis=1).mean()),
        'mean_gripper_abs_difference': float(np.abs(d[:, 7]).mean()),
        'mean_action_norm_reference': float(ref.mean()),
        'first_8d_action_l2': float(norms[0]),
    }


def analyze(doc: dict, actions: dict, pairs: dict) -> dict:
    records = []
    original = {(p['episode']): p for p in doc.get('pairs', [])}
    for ep in sorted(pairs):
        a, b = actions[(ep, 0)], actions[(ep, 1)]
        x = pairs[ep]
        original_pair = original.get(ep)
        if original_pair is None:
            raise ValueError('MISSING_ORIGINAL_PAIR')
        original_l2 = original_pair.get('mean_action_l2_first_common_steps')
        all_mean = range_metrics(a, b, min(len(a), len(b)))['mean_8d_action_l2']
        if original_l2 is None or not math.isclose(float(original_l2), all_mean, abs_tol=1e-4):
            raise ValueError('ACTION_METRIC_DOES_NOT_REPRODUCE_ORIGINAL')
        by_id = {trial['source_train_demo_id']: trial for trial in x}
        records.append({
            'episode': ep,
            'source_ids': [0, 1],
            'successes_by_source': [bool(by_id[i]['success']) for i in EXPECTED_IDS],
            'matched_reset_seed': by_id[0]['reset_seed'] == by_id[1]['reset_seed'],
            'matched_policy_rng_seed': by_id[0]['policy_rng_seed'] == by_id[1]['policy_rng_seed'],
            'mean_8d_action_l2_original': float(original_l2),
            'first_action_8d_delta': (a[0] - b[0]).tolist(),
            'prefix_metrics': {str(min(prefix, len(a), len(b))): range_metrics(a, b, prefix)
                               for prefix in PREFIXES},
        })
    return {
        'status': 'READ_ONLY_DIAGNOSIS_NOT_NEW_ROLLOUT',
        'task': EXPECTED_TASK,
        'root': str(Path(doc['trials'][0]['actions_path']).resolve().parents[2]),
        'real_rollouts': 4,
        'successes': sum(bool(x['success']) for x in doc['trials']),
        'pairs': records,
        'causal_video_necessity_proven': False,
        'stage_failure_classification_without_video': 'UNKNOWN_REQUIRES_VISUAL_REVIEW',
        'note': 'Early 1-step / 20-step action differences are more diagnostic than total 800-step L2, but still require verification of deterministic initial observations.',
    }


def make_montage(root: Path, pairs: dict, out: Path) -> dict:
    """Representative synchronized frames from original MP4s; no inference on stages."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageOps

    cell_w, cell_h = 158, 138
    header_w, band_h = 175, 32
    count = len(STORY_STEPS)
    width = header_w + count * cell_w
    height = 4 * (cell_h + band_h) + 38
    canvas = Image.new('RGB', (width, height), '#fff')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), 'StackCups DEV: official robot rollout frames (steps, not stages)', fill='#111')
    idx = 0
    notes = []
    for ep in sorted(pairs):
        for source in EXPECTED_IDS:
            trial = next(x for x in pairs[ep] if x['source_train_demo_id'] == source)
            root_mp4 = root / f'episode_{ep:02d}' / f'source_{source:04d}' / 'rollout.mp4'
            if not root_mp4.is_file() or trial.get('video_error'):
                notes.append(f'Video missing/broken for episode={ep} source={source}')
                idx += 1
                continue
            # Decode sequentially once; keep only the 10 requested frames in memory.
            selected = {min(step, int(trial['steps'])): None for step in STORY_STEPS}
            reader = imageio.get_reader(str(root_mp4), format='ffmpeg')
            try:
                for t, frame in enumerate(reader):
                    if t in selected:
                        selected[t] = Image.fromarray(frame[:, :, :3]).convert('RGB')
                    if t >= max(selected):
                        break
            finally:
                reader.close()
            row_y = 38 + idx * (cell_h + band_h)
            draw.text((10, row_y + 4), f'Episode {ep}', fill='#222')
            draw.text((10, row_y + 20), f'TRAIN {source:04d}', fill='#333')
            for j, step in enumerate(STORY_STEPS):
                chosen=min(step, int(trial['steps']))
                x = header_w + j * cell_w
                img = selected.get(chosen)
                if img is not None:
                    thumbnail = ImageOps.contain(img, (cell_w - 6, cell_h - 4))
                    canvas.paste(thumbnail, (x + 3 + (cell_w - 6 - thumbnail.width) // 2, row_y + (cell_h - thumbnail.height)//2))
                else:
                    draw.text((x + 7, row_y + 50), 'NO FRAME', fill='#a33')
                    notes.append(f'Frame {chosen} missing episode={ep} source={source}')
                draw.text((x + 5, row_y + cell_h + 5), f'step {chosen}', fill='#333')
            idx += 1
    canvas.save(out, quality=91)
    return {'montage_path': str(out), 'video_read_warnings': notes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=Path('outputs/roboreel_video_dependency_dev/stack_v1'))
    parser.add_argument('--out', type=Path,
        default=Path('outputs/roboreel_video_dependency_diagnosis/stack_v1'))
    parser.add_argument('--no-montage', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('OUTPUT_MUST_BE_SEPARATE_FROM_FROZEN_EXPERIMENT')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('REFUSE_TO_OVERWRITE_EXISTING_DIAGNOSIS: ' + str(out))
    doc, actions, pairs = validate_source(root)
    report = analyze(doc, actions, pairs)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_montage:
        try:
            report['video_montage'] = make_montage(root, pairs, out / 'rollout_montage.jpg')
        except Exception as exc:
            report['video_montage'] = {'error': f'{type(exc).__name__}: {str(exc)[:350]}',
                                       'original_video_unchanged': True}
    (out / 'diagnosis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# Read-only StackCups video-conditioned policy diagnosis', '',
        f'- Source: `{root}`',
        f'- Successful real rollouts: **{report["successes"]}/4**.',
        '- No new PhysX, policy training, or video features were executed.',
        '- Stage labels MUST be assigned manually by viewing the montage/original MP4s.',
        '- First 1-step and 20-step action differences do not by themselves prove human-video necessity.',
        '',
        '| Episode | First-step 8D L2 | First-20 8D mean L2 | Full 800 8D mean L2 |',
        '|---|---:|---:|---:|',
    ]
    for pair in report['pairs']:
        m = pair['prefix_metrics']
        full = m[str(max(map(int, m)))]
        lines.append(f'| {pair["episode"]} | {m["1"]["first_8d_action_l2"]:.6f} | '
                     f'{m["20"]["mean_8d_action_l2"]:.6f} | '
                     f'{full["mean_8d_action_l2"]:.6f} |')
    lines += ['', '- Upload `rollout_montage.jpg` for visual failure-stage review.', '']
    (out / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    compact = {'status':report['status'],'successes':report['successes'],
               'first_8d_l2':[p['prefix_metrics']['1']['first_8d_action_l2'] for p in report['pairs']],
               'first20_mean_8d_l2':[p['prefix_metrics']['20']['mean_8d_action_l2'] for p in report['pairs']],
               'full_mean_8d_l2':[p['mean_8d_action_l2_original'] for p in report['pairs']],
               'summary':str(out / 'SUMMARY.md'),
               'montage':str(out / 'rollout_montage.jpg') if (out/'rollout_montage.jpg').exists() else None}
    print('ROBOREEL_STACK_READONLY_DIAGNOSIS',json.dumps(compact, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
