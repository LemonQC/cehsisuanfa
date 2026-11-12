#!/usr/bin/env python3
"""M1B3 source: bounded, step-free frozen-state planning audit.

Run --inventory first.  --plan constructs the original ManiSkill scene and calls
MPLib planning-only APIs; NEVER calls env.step()/follow_path(), saves no new
trajectory and does not alter any existing physical evidence.

No plan status is a real physical grasp certificate.  The original failed
``grasp_pose`` quaternion was not serialized; it is reconstructed by the
frozen generator and identified as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CASE = Path('outputs/m1_source_geom_aligned_topdown_v1/source_aligned_topdown')
REPORT = Path('outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_audit.json')
INVENTORY = Path('outputs/m1_source_geom_aligned_topdown_v1/frozen_planning_inventory.json')
PHYSICAL_STEPS = 65


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_json(case):
    # Reuse strict genuine-result gate rather than trusting a merely named JSON.
    from scripts.m1_source_65_step_force_vs_planner_audit import analyse
    old = analyse(case)
    if old['actual_physx_new_runs'] != 0 or old['grasp_pose_executed_actions'] != 0:
        raise ValueError('Frozen source status contradicts 65-action planner failure')
    st = json.loads((case / 'status.json').read_text())
    trace = json.loads((case / 'physical_trace.json').read_text())
    return st, trace[-1], old


def h5_inventory(path):
    import h5py
    with h5py.File(path, 'r') as f:
        results = []
        def visit(name, node):
            if hasattr(node, 'shape'):
                results.append({'path': name, 'shape': list(node.shape), 'dtype': str(node.dtype)})
        f.visititems(visit)
        return results[:150]


# Explicitly documented ManiSkill articulation state format:
#   root p(3), root quaternion wxyz(4), root linear vel(3), root angular vel(3),
#   active-joint qpos(DOF), active-joint qvel(DOF).
# Original Panda has 9 active DoF: 13 + 9 + 9 = 31.
# See ManiSkill Articulation.get_state()/set_state(), NOT the flat obs schema.
PACKED_PANDA_PATH = 'traj_0/env_states/articulations/panda'
PANDA_QPOS = slice(13, 22)
PANDA_QVEL = slice(22, 31)


def candidate_datasets(h5):
    """Accept named qpos or the original ManiSkill 31D Panda state only."""
    items = []
    def visit(name, node):
        if not hasattr(node, 'shape'):
            return
        shape = tuple(int(x) for x in node.shape)
        if name.lower().endswith('/qpos') and (
                shape == (PHYSICAL_STEPS + 1, 9)
                or shape == (PHYSICAL_STEPS + 1, 1, 9)):
            items.append((name, shape))
        elif name == PACKED_PANDA_PATH and shape == (PHYSICAL_STEPS + 1, 31):
            items.append((name, shape))
    h5.visititems(visit)
    return items


def decode_panda_state(packed):
    """Verified ManiSkill articulation layout; reject wrong shape/quat/data.

    This does NOT decode robot qpos from flat observations: it only accepts
    the named env_states/articulations/panda field as recorded by ManiSkill.
    """
    import numpy as np
    arr = np.asarray(packed, dtype=np.float64)
    if arr.shape != (PHYSICAL_STEPS + 1, 31):
        raise ValueError('PANDA_ARTICULATION_WRONG_SHAPE_NOT_66x31')
    if not np.isfinite(arr).all():
        raise ValueError('PANDA_ARTICULATION_NONFINITE')
    # Quaternions in articulation get_state are root wxyz, not angles.
    norms = np.linalg.norm(arr[:, 3:7], axis=1)
    if not np.all(np.abs(norms - 1.0) <= 0.02):
        raise ValueError('PANDA_ARTICULATION_ROOT_QUATERNION_NOT_UNIT')
    joints = arr[:, PANDA_QPOS]
    velocities = arr[:, PANDA_QVEL]
    if joints.shape != (66, 9) or velocities.shape != (66, 9):
        raise ValueError('PANDA_ARTICULATION_DOF_MISMATCH')
    # Panda has 7 revolute joints and two nonnegative finger prismatic joints.
    # Only loose sanity checks; exact joint limits come from local robot model.
    if (np.abs(joints[:, :7]).max() > 4.0 or
            np.any(joints[:, -2:] < -0.005) or
            np.any(joints[:, -2:] > 0.046)):
        raise ValueError('PANDA_ARTICULATION_JOINT_SANITY_FAILED')
    return joints.copy(), velocities.copy()


def extract_snapshot(case, explicit_h5=None, dataset=None):
    import h5py
    import numpy as np
    matches = [Path(explicit_h5)] if explicit_h5 else sorted(case.glob('*.h5'))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError('Exactly one original source H5 required; inventory and set --h5 (no guessed file)')
    path = matches[0]
    with h5py.File(path, 'r') as f:
        # 65 physically executed control actions => 66 saved states.
        if 'traj_0/actions' not in f or tuple(f['traj_0/actions'].shape) != (65, 8):
            raise ValueError('SOURCE_ACTIONS_MISSING_OR_NOT_65x8')
        found = candidate_datasets(f)
        if dataset:
            if dataset not in [name for name, _ in found]:
                raise ValueError('Selected dataset unavailable or not verified 66x9/66x31 Panda: '+dataset)
            chosen = dataset
        else:
            if not found:
                raise ValueError('NO_QPOS_DATASET: no verified 66x9 qpos or original 66x31 Panda articulation state')
            chosen = (PACKED_PANDA_PATH if PACKED_PANDA_PATH in [n for n,_ in found]
                      else found[0][0])
        decoded = {}
        for name, shape in found:
            if name == PACKED_PANDA_PATH:
                raw31 = np.asarray(f[name], dtype=np.float64)
                qpos, qvel = decode_panda_state(raw31)
                decoded[name] = qpos
            else:
                qpos = np.asarray(f[name], dtype=np.float64).reshape(PHYSICAL_STEPS+1, 9)
                if not np.isfinite(qpos).all():
                    raise ValueError('Non-finite saved robot qpos in '+name)
                decoded[name] = qpos
        if any(not np.allclose(decoded[chosen], value, atol=1e-5, rtol=0)
               for key, value in decoded.items() if key != chosen):
            raise ValueError('AMBIGUOUS_QPOS: named and packed state 66x9 joint trajectories disagree')
        raw = decoded[chosen]
        q = raw[-1]
        packed_available = PACKED_PANDA_PATH in f and (
            tuple(f[PACKED_PANDA_PATH].shape) == (PHYSICAL_STEPS+1, 31))
        # Even if named qpos is selected, validate and expose packed state if present.
        full_state = (np.asarray(f[PACKED_PANDA_PATH], dtype=np.float64)
                      if packed_available else None)
        if full_state is not None:
            decode_panda_state(full_state)
        actor_snapshots = {}
        for actor_name in ('cube', 'm1b3_dynamic_blocker'):
            apath = 'traj_0/env_states/actors/' + actor_name
            if apath in f:
                actor = np.asarray(f[apath], dtype=np.float64)
                if actor.shape != (PHYSICAL_STEPS+1, 13) or not np.isfinite(actor).all():
                    raise ValueError('Saved actor state malformed: '+apath)
                actor_snapshots[actor_name] = actor[-1, :3].tolist()
        return {
            'h5_path': str(path.resolve()), 'h5_sha256': sha256(path),
            'dataset': chosen, 'available_datasets': found,
            'qpos_9': q.tolist(), 'qpos_previous_9': raw[-2].tolist(),
            'qpos_snapshot_index': PHYSICAL_STEPS,
            'qpos_66_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'packed_articulation_state_31': (None if full_state is None
                                              else full_state[-1].tolist()),
            'packed_qvel_9': (None if full_state is None
                              else full_state[-1, PANDA_QVEL].tolist()),
            'actor_last_xyz': actor_snapshots,
            'layout': ('MANISKILL_ARTICULATION_ROOT13_QPOS9_QVEL9' if full_state is not None
                       else 'NAMED_EXPLICIT_QPOS9_ONLY'),
            'layout_source': ('ManiSkill Articulation.get_state/set_state: 13+2*DOF; '
                              'root pose7, root linear3, root angular3, qpos9, qvel9'),
        }


def verdict(ik_status, self_col, env_col, screw_status, certified_exact_pose=False):
    """Conservative: absence of collision evidence != absence of collision."""
    if ik_status != 'Success':
        return 'INCONCLUSIVE_IK_OR_COLLISION_FAILURE'
    if self_col is True or env_col is True:
        return 'GOAL_COLLISION_REPORTED_BY_PLANNER'
    if self_col is None or env_col is None:
        return 'INCONCLUSIVE_COLLISION_API'
    if screw_status != 'Success':
        return 'IK_CANDIDATE_BUT_SCREW_REJECTED'
    # The original pose was not serialized, so this isn't a bitwise-exact check.
    if not certified_exact_pose:
        return 'PLANNING_CANDIDATE_RECONSTRUCTED_POSE_ONLY'
    return 'PLANNING_CANDIDATE_NOT_PHYSICAL_VALIDATION'


def collision_query(planner, candidate, mplib_version):
    """Check *candidate* qpos using the exact MPLib versioned signatures.

    MPLib 0.1.1: check_for_self_collision(articulation=None, qpos=None)
                 check_for_env_collision(articulation=None, qpos=None,
                                         with_point_cloud=False, use_attach=False)
    Here passing qpos as the FIRST positional argument would instead be
    interpreted as an articulation, yielding false conclusions/exceptions.
    Newer MPLib releases accept the qpos as the first positional argument.
    """
    legacy = str(mplib_version) == '0.1.1'
    out = {}
    for key, method in [('self', 'check_for_self_collision'),
                        ('environment', 'check_for_env_collision')]:
        fn = getattr(planner, method, None)
        if not callable(fn):
            out[key] = {'observed': None, 'reason': 'API_UNAVAILABLE'}
            continue
        try:
            if legacy:
                if key == 'environment':
                    # Original source uses a 792-pt static+blocker cloud.
                    # By default, v0.1.1 omits that cloud from env collisions.
                    hits = fn(qpos=candidate, with_point_cloud=True,
                              use_attach=False)
                else:
                    hits = fn(qpos=candidate)
            else:
                hits = fn(candidate)
            if hits is None or not hasattr(hits, '__len__'):
                raise ValueError('Missing collision result sequence')
            out[key] = {
                'observed': bool(len(hits)), 'count': len(hits),
                'details': [str(x)[:350] for x in list(hits)[:8]],
                'api': ('mplib_0.1.1_explicit_qpos'
                        if legacy else 'mplib_modern_positional_qpos'),
                'with_point_cloud': key == 'environment',
            }
        except Exception as exc:
            out[key] = {
                'observed': None,
                'reason': f'{type(exc).__name__}: {exc}'[:350],
                'api': 'mplib_0.1.1_explicit_qpos' if legacy
                       else 'mplib_modern_positional_qpos',
            }
    return out


def make_mplib_ik_pose(world_pose_xyz_wxyz):
    """Version-accurate input to Planner.IK, never invent a Pose type.

    MPLib 0.1.1 source (haosulab/MPlib tag v0.1.1, mplib/planner.py):
        IK(goal_pose: [x,y,z,qw,qx,qy,qz], start_qpos, ...)
        plan_screw(target_pose: same 7D array, qpos, ..., use_point_cloud)
    It exports NO Pose class. MPLib >=0.2 instead uses a C++ Pose binding.
    Unknown versions lacking both documented constructors fail closed.
    """
    import importlib
    import numpy as np

    values = np.asarray(world_pose_xyz_wxyz, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('MPLIB_IK_POSE_EXPECTED_FINITE_7D_XYZ_WXYZ')
    if abs(float(np.linalg.norm(values[3:])) - 1.) > .02:
        raise ValueError('MPLIB_IK_POSE_QUATERNION_NOT_UNIT')

    import mplib
    version = str(getattr(mplib, '__version__', 'UNKNOWN'))
    if version == '0.1.1':
        return values.copy(), 'mplib_0.1.1_flat_7d_xyz_wxyz'

    # Keep documented 0.2+ path where Pose is a pybind type. No guesswork
    # involving SAPIEN Pose, whose bindings may not match the planner.
    errors = []
    for name in ('mplib.pymp', 'mplib'):
        try:
            module = importlib.import_module(name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f'{name}: {type(exc).__name__}')
            continue
        ctor = getattr(module, 'Pose', None)
        if ctor is None:
            errors.append(f'{name}: Pose unavailable')
            continue
        try:
            return ctor(p=values[:3], q=values[3:]), f'{name}.Pose'
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(f'{name}.Pose: {type(exc).__name__}: {str(exc)[:160]}')
    raise RuntimeError(
        'MPLIB_IK_POSE_API_UNAVAILABLE: version=' + version
        + ' module=' + str(getattr(mplib, '__file__', 'unknown'))
        + ' attempted=' + '; '.join(errors))


class NoPhysicalStep:
    """Deny path following even if a wrapper accidentally invokes it."""
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.action_space = env.action_space
    def __getattr__(self, key):
        return getattr(self.env, key)
    def step(self, action):
        raise RuntimeError('FORBIDDEN_NEW_PHYSX_STEP_IN_OFFLINE_PLANNING_GATE')


def planning_only(snapshot, last, official_root):
    """One original-scene, original-pose planning evaluation; no trajectories."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import numpy as np
    import torch
    import mplib
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb)
    from agcofl.m1b3_corridor_physics import register_env, SEED, MAX_STEPS
    from agcofl.m1_source_geom_aligned_topdown_v1 import original_obstacle_points
    from agcofl.m1_current_executor_swap import get_planner

    if not np.allclose(snapshot['qpos_9'][-2:], last['finger_joint_qpos_m'], atol=.003, rtol=0):
        raise ValueError('Saved H5 qpos fingers do not match frozen 65th physical state')
    if snapshot['packed_articulation_state_31'] is None:
        raise ValueError('FULL_ARTICULATION_STATE_REQUIRED_FOR_PLAN: qpos alone does not verify root/qvel')
    actor_xyz = snapshot['actor_last_xyz']
    for actor, recorded_key in [('cube', 'task_cube'), ('m1b3_dynamic_blocker', 'blocker')]:
        if actor not in actor_xyz or np.linalg.norm(
                np.asarray(actor_xyz[actor]) - np.asarray(last[recorded_key])) > .001:
            raise ValueError('Saved H5 last actor state disagrees with 65th trace: '+actor)
    env = None
    solver = None
    try:
        env_id = register_env(official_root)
        env = gym.make(env_id, obs_mode='state', robot_uids='panda',
                       num_envs=1, control_mode='pd_joint_pos',
                       render_mode='rgb_array', sim_backend='physx_cpu',
                       distract=False, max_episode_steps=MAX_STEPS)
        env.reset(seed=SEED, options={'m1b3_open': False})
        b = env.unwrapped
        q = np.asarray(snapshot['qpos_9'], dtype=np.float64)
        full = np.asarray(snapshot['packed_articulation_state_31'], dtype=np.float64)
        b.agent.robot.set_state(torch.as_tensor(full[None, :], device=b.device, dtype=torch.float32))
        actual_full = b.agent.robot.get_state().detach().cpu().numpy().reshape(-1)
        if actual_full.shape != (31,) or not np.allclose(actual_full, full, atol=2e-5, rtol=0):
            raise ValueError('Local ManiSkill articulation layout differs from saved frozen 31D state')
        actual_q = b.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        if actual_q.shape != (9,) or not np.allclose(actual_q, q, atol=2e-5, rtol=0):
            raise ValueError('Failed to restore frozen robot qpos')
        actual_tcp = b.agent.tcp_pose.p.detach().cpu().numpy().reshape(-1)[:3]
        if float(np.linalg.norm(actual_tcp - np.asarray(last['tcp']))) > .005:
            raise ValueError('Saved H5 qpos FK disagrees with last measured TCP >5mm')
        cube = b.obj.pose.p.detach().cpu().numpy().reshape(-1)[:3]
        blocker = b.blocker.pose.p.detach().cpu().numpy().reshape(-1)[:3]
        if (np.linalg.norm(cube - np.asarray(last['task_cube'])) > .001
                or np.linalg.norm(blocker - np.asarray(last['blocker'])) > .001):
            raise ValueError('Reset scene object/blocker mismatch >1mm vs frozen last step')

        # EXACT frozen source code recipe; not a guessed hand-written quaternion.
        approaching = np.array([0., 0., -1.], dtype=float)
        closing_target = np.array([1., 0., 0.], dtype=float)
        gi = compute_grasp_info_by_obb(get_actor_obb(b.obj),
                                       approaching=approaching,
                                       target_closing=closing_target, depth=.025)
        closing = np.asarray(gi['closing'], dtype=float)
        if abs(float(closing @ closing_target)) < .98:
            raise ValueError('Frozen OBB closing direction changed')
        target = b.agent.build_grasp_pose(approaching, closing, b.obj.pose.sp.p)
        pose7 = np.concatenate([np.asarray(target.p), np.asarray(target.q)]).astype(float)
        if pose7.shape != (7,) or not np.isfinite(pose7).all():
            raise ValueError('Non-finite reconstructed grasp pose')
        solver = get_planner(NoPhysicalStep(env))
        pts = original_obstacle_points()
        solver.add_collision_pts(pts)
        solver.use_point_cloud = True
        inner = solver.planner
        q_start = b.agent.robot.get_qpos().detach().cpu().numpy()[0]

        # NOTE: high-level IK may itself filter collisions; its failure is NOT
        # a proof that raw forward kinematics are incapable of reaching pose.
        ik_pose, ik_pose_factory = make_mplib_ik_pose(pose7)
        ik_raw = inner.IK(ik_pose, q_start, n_init_qpos=20)
        if not isinstance(ik_raw, tuple) or len(ik_raw) != 2:
            raise RuntimeError('Pinned MPLib IK API changed; no automatic fallback')
        ik_status, solutions = ik_raw
        if solutions is None:
            candidates = []
        elif isinstance(solutions, np.ndarray) and solutions.ndim == 1:
            candidates = [solutions]
        else:
            candidates = list(solutions)
        col = {'self': {'observed': None, 'reason': 'NO_IK_SOLUTION'},
               'environment': {'observed': None, 'reason': 'NO_IK_SOLUTION'}}
        candidate = None
        if str(ik_status) == 'Success' and candidates:
            candidate = np.asarray(candidates[0], dtype=float).reshape(-1)
            col = collision_query(inner, candidate, getattr(mplib, '__version__', 'UNKNOWN'))

        screw_raw = inner.plan_screw(pose7, q_start,
                                     time_step=b.control_timestep,
                                     use_point_cloud=True)
        if not isinstance(screw_raw, dict):
            raise RuntimeError('Pinned MPLib plan_screw returned unsupported result')
        screw_status = str(screw_raw.get('status', 'STATUS_MISSING'))
        output = {
            'reconstructed_grasp_pose_world_xyz_wxyz': pose7.tolist(),
            'pose_provenance': 'REGENERATED_FROM_FROZEN_CODE_NOT_SERIALIZED_IN_FAILED_RUN',
            'restored_qpos_9': q_start.tolist(),
            'restored_articulation_root13': actual_full[:13].tolist(),
            'articulation_state_layout_validated_against_local_mani_skill': True,
            'restored_tcp_world_xyz_m': actual_tcp.tolist(),
            'restored_object_world_xyz_m': cube.tolist(),
            'restored_blocker_world_xyz_m': blocker.tolist(),
            'collision_points_count': int(len(pts)),
            'collision_points_not_exact_mesh': True,
            'mplib_version': getattr(mplib, '__version__', 'UNKNOWN'),
            'mplib_api_contract': ('0.1.1_FLAT7_IK_AND_EXPLICIT_QPOS_COLLISION'
                                  if str(getattr(mplib, '__version__', '')) == '0.1.1'
                                  else 'MODERN_POSE_IK_COLLISION'),
            'ik_pose_factory': ik_pose_factory,
            'ik_status': str(ik_status), 'ik_solutions_returned': len(candidates),
            'first_ik_candidate': None if candidate is None else candidate.tolist(),
            'ik_collision_checks': col,
            'screw_status': screw_status,
            'screw_path_computed_no_execution': screw_status == 'Success',
        }
        output['decision'] = verdict(
            str(ik_status), col['self']['observed'],
            col['environment']['observed'], screw_status, False)
        return output
    finally:
        if solver is not None:
            solver.close()
        if env is not None:
            env.close()


def write_once(path, result):
    content = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if path.is_file():
        if path.read_text() != content:
            raise FileExistsError('Existing frozen audit differs, preserving original: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', type=Path, default=CASE)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--h5', type=Path)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--official-root', default='third_party/roboreel_models')
    ap.add_argument('--plan', action='store_true', help='Step-free original scene + frozen MPLib planning; NO rollout')
    a = ap.parse_args(argv)
    try:
        st, last, old = frozen_json(a.case)
        snap = extract_snapshot(a.case, a.h5, a.dataset)
        if any(abs(snap['qpos_9'][-2+i] - last['finger_joint_qpos_m'][i]) > .003
               for i in range(2)):
            raise ValueError('Final finger positions mismatch real trace')
        for actor, trace_key in [('cube', 'task_cube'), ('m1b3_dynamic_blocker', 'blocker')]:
            xyz = snap['actor_last_xyz'].get(actor)
            if xyz is not None and math.dist(xyz, last[trace_key]) > .001:
                raise ValueError('Saved H5 actor position differs >1mm from frozen trace: '+actor)
        result = {
            'type': 'M1B3_SOURCE_FROZEN_PLANNING_ONLY',
            'source_case': str(a.case.resolve()),
            'original_physx_actions': PHYSICAL_STEPS,
            'new_physx_actions': 0,
            'fresh_scene_initialization_may_use_physics_engine': bool(a.plan),
            'original_status_sha256': old['status_sha256'],
            'original_trace_sha256': old['physical_trace_sha256'],
            'snapshot': snap,
            'frozen_actual_tcp_xyz_m': last['tcp'],
            'mode': 'PLANNING_ONLY' if a.plan else 'INVENTORY_ONLY',
            'planning': None,
            'decision': 'INVENTORY_DONE_PLANNING_NOT_RUN',
            'authorizes_physics': False,
            'research_status': 'SAME_GOAL_SOURCE_NOT_VALIDATED; M2_NOT_RUN',
        }
        if a.plan:
            result['planning'] = planning_only(snap, last, a.official_root)
            result['decision'] = result['planning']['decision']
        # Existing physical files can never be overwritten or modified.
        if (sha256(a.case / 'status.json') != old['status_sha256'] or
                sha256(a.case / 'physical_trace.json') != old['physical_trace_sha256'] or
                sha256(Path(snap['h5_path'])) != snap['h5_sha256']):
            raise RuntimeError('Frozen inputs changed during read-only diagnostic')
        out = a.out if a.out is not None else (REPORT if a.plan else INVENTORY)
        write_once(out, result)
        print('M1_SOURCE_FROZEN_PLANNING_GATE', json.dumps({
            'decision': result['decision'], 'mode': result['mode'],
            'new_physx_actions': 0, 'authorizes_physics': False,
            'snapshot_dataset': snap['dataset'], 'decoded_layout': snap['layout'],
            'finger_qpos_m': snap['qpos_9'][-2:], 'report': str(out)
        }, ensure_ascii=False), flush=True)
    except Exception as exc:
        # Error path deliberately does not fabricate a classification or a success.
        evidence_inventory = None
        candidates = [a.h5] if a.h5 else sorted(a.case.glob('*.h5'))
        if len(candidates) == 1 and candidates[0].is_file():
            try:
                evidence_inventory = h5_inventory(candidates[0])
            except Exception as inventory_exc:
                evidence_inventory = f'UNAVAILABLE: {type(inventory_exc).__name__}: {inventory_exc}'
        print('M1_SOURCE_FROZEN_PLANNING_INCONCLUSIVE', json.dumps({
            'decision': 'INCONCLUSIVE', 'reason': f'{type(exc).__name__}: {exc}'[:700],
            'new_physx_actions': 0, 'authorizes_physics': False,
            'h5_dataset_inventory': evidence_inventory,
        }, ensure_ascii=False), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
