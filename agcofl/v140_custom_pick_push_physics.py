"""CUSTOM-ADDITIONAL same-goal Pick-source -> Push-only Panda physics feasibility.

This is NOT an official RoboReel task, and NOT a learned method. Source and
target share a single custom PushCube goal, obstacle scene, reset seed and
success predicate. A source must actually grasp/lift/release in PhysX before
its effect trace is accepted; a failed source is never a valid demonstration.
Waypoint/geometry proposals are NEGATIVE CONTROLS, not AG-COFL innovation.
No GPU physical trials have been run by this commit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .roboreel_bridge import DiagnosticPushTeacher, delta_pose_action, np1

CUSTOM_ID = "AGCOFLSameGoalPushCube-v0"
WALL_OFFSET_X = 0.075
WALL_STEP_Y = 0.05
WALL_HALF = 0.023
METHODS = ("goal_only", "waypoint_oracle", "raw_source_xy")


def success_flag(info):
    return bool(np1(info.get("success", False))[0])


def route_via(start, side=1):
    """Grounded waypoints for a static obstacle, NOT a learned projection."""
    x, y = map(float, start[:2])
    return [[x + 0.025, y + side * 0.17],
            [x + 0.125, y + side * 0.17]]


def source_valid(summary):
    """Hard gate on actually observed contact, lift, AND same-task success."""
    return (summary["env_success"] and summary["ever_grasped"]
            and summary["peak_lift_m"] >= 0.055
            and summary["released_on_table"]
            and summary["final_goal_error_m"] < 0.05)


def register_custom_env(official):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien
    from mani_skill.utils.building import actors
    from mani_skill.utils.structs import Pose
    sys.path.insert(0, str((Path(official).resolve() / "roboreel_env/src")))
    import envs  # noqa: F401
    from envs.push_cube import PushCubeEnvV2

    class SameGoalObstacleEnv(PushCubeEnvV2):
        """Same physical cube, tabletop and XY goal for Pick and Push modes."""
        def _load_scene(self, options):
            super()._load_scene(options)
            self.physics_walls = [
                actors.build_cube(
                    self.scene, half_size=WALL_HALF, body_type="static",
                    color=[0.85, 0.2, 0.2, 1.0],
                    name=f"capability_wall_{i}",
                    initial_pose=sapien.Pose(p=[0.5, 0.5, WALL_HALF]))
                for i in range(-2, 3)
            ]

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            start = np1(self.obj.pose.p)[:3]
            for i, wall in zip(range(-2, 3), self.physics_walls):
                x = float(start[0] + WALL_OFFSET_X)
                y = float(start[1] + i * WALL_STEP_Y)
                pose = Pose.create_from_pq(
                    p=np.asarray([[x, y, WALL_HALF]], np.float32),
                    q=[1, 0, 0, 0])
                wall.set_pose(pose)

    if CUSTOM_ID not in gym.registry:
        gym.register(id=CUSTOM_ID, entry_point=SameGoalObstacleEnv,
                     max_episode_steps=800)
    return CUSTOM_ID


def make_env(official, seed):
    import gymnasium as gym
    register_custom_env(official)
    env = gym.make(
        CUSTOM_ID, obs_mode="state", num_envs=1, robot_uids="panda",
        control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
        render_mode=None, render_backend="none", distract=False,
        max_episode_steps=800)
    env.reset(seed=seed)
    b = env.unwrapped
    start = np1(b.obj.pose.p)[:3].astype(float)
    goal = np1(b.goal_pos)[:3].astype(float)
    return env, start, goal


class ScriptedPickSource:
    """Physical feasibility source, NOT a human video or trained source policy."""
    def __init__(self, env, start, goal):
        self.b = env.unwrapped
        self.action_shape = env.action_space.shape
        self.start, self.goal = start, goal
        self.phase = "above"
        self.steps_in_phase = 0
        self.trace = []

    def _to(self, phase):
        self.phase = phase
        self.steps_in_phase = 0

    def act(self):
        b = self.b
        obj = np1(b.obj.pose.p)[:3]
        tcp = np1(b.agent.tcp_pose.p)[:3]
        grasp = bool(np1(b.agent.is_grasping(b.obj))[0])
        close = self.phase not in ("above", "descend", "release", "done", "failed")
        pos = np.r_[self.start[:2], 0.14]
        if self.phase == "above":
            if np.linalg.norm(tcp - pos) < .035:
                self._to("descend")
        elif self.phase == "descend":
            pos = np.r_[self.start[:2], self.start[2]]
            if np.linalg.norm(tcp - pos) < .02:
                self._to("grasp")
        elif self.phase == "grasp":
            pos = np.r_[self.start[:2], self.start[2]]
            close = True
            if grasp:
                self._to("lift")
            elif self.steps_in_phase > 45:
                self._to("failed")
        elif self.phase == "lift":
            pos = np.r_[self.start[:2], .16]
            if obj[2] > self.start[2] + .065:
                self._to("habitual_detour")
        elif self.phase == "habitual_detour":
            pos = np.r_[self.start[0] - .035, self.start[1] - .12, .16]
            if np.linalg.norm(tcp - pos) < .045:
                self._to("over_goal")
        elif self.phase == "over_goal":
            pos = np.r_[self.goal[:2], .16]
            if np.linalg.norm(tcp - pos) < .04:
                self._to("lower")
        elif self.phase == "lower":
            pos = np.r_[self.goal[:2], self.start[2] + .02]
            if np.linalg.norm(tcp - pos) < .027:
                self._to("release")
        elif self.phase == "release":
            pos = np.r_[self.goal[:2], .055]
            close = False
            if self.steps_in_phase > 30:
                self._to("done")
        elif self.phase in ("done", "failed"):
            pos = tcp
            close = False
        else:
            raise AssertionError(self.phase)
        act = delta_pose_action(tcp, pos, self.action_shape).reshape(-1)
        act[-1] = 1. if close else -1.
        self.steps_in_phase += 1
        self.trace.append(dict(phase=self.phase, cube=obj.tolist(),
                               tcp=tcp.tolist(), is_grasped=grasp,
                               gripper_command=float(act[-1])))
        return act.reshape(self.action_shape)


class RoutePush(DiagnosticPushTeacher):
    """Oracle-state controller; the only change is subgoal selection."""
    def __init__(self, env, via):
        super().__init__(env, max_attempts=2)
        self.via = [np.asarray(p, dtype=np.float32) for p in via]
        self.route_index = 0
        self.visited = []

    def goal_xyz(self):
        if self.route_index < len(self.via):
            return np.r_[self.via[self.route_index], self.cube_xyz()[2]]
        return super().goal_xyz()

    def act(self):
        if self.route_index < len(self.via):
            d = float(np.linalg.norm(
                self.cube_xyz()[:2] - self.via[self.route_index]))
            if d < .045:
                self.visited.append(self.route_index)
                self.route_index += 1
                self._phase("above_contact")
                self.attempt = 0
                self.stale_steps = 0
        action = super().act()
        # Enforce push-only actuation: no closing gripper, no grasp lifting.
        action = np.asarray(action, dtype=np.float32).copy()
        action.reshape(-1)[-1] = -1.
        return action


def run_source(official, seed, out, max_steps):
    env, start, goal = make_env(official, seed)
    source = ScriptedPickSource(env, start, goal)
    grasped = False
    zpeak = float(start[2])
    done = False
    try:
        for _ in range(max_steps):
            action = source.act()
            _, _, terminated, truncated, info = env.step(action)
            grasped |= bool(np1(env.unwrapped.agent.is_grasping(env.unwrapped.obj))[0])
            zpeak = max(zpeak, float(np1(env.unwrapped.obj.pose.p)[2]))
            done = success_flag(info)
            if source.phase == "done" or source.phase == "failed" or (
                bool(np1(truncated)[0]) or bool(np1(terminated)[0])):
                break
        obj = np1(env.unwrapped.obj.pose.p)[:3]
        grasp_now = bool(np1(env.unwrapped.agent.is_grasping(env.unwrapped.obj))[0])
        goal_dist = float(np.linalg.norm(obj[:2] - goal[:2]))
        # The goal predicate is checked AFTER source release, never during lift.
        actual_success = bool(np1(env.unwrapped.evaluate()["success"])[0])
        summary = dict(type="SIMULATED_PICK_SOURCE_PHYSICS_ONLY",
                       seed=seed, initial_cube=start.tolist(),
                       goal=goal.tolist(), steps=len(source.trace),
                       ever_grasped=grasped,
                       peak_lift_m=zpeak - float(start[2]),
                       released_on_table=not grasp_now and obj[2] < start[2] + .006,
                       final_goal_error_m=goal_dist, env_success=actual_success,
                       end_phase=source.phase, final_cube=obj.tolist(),
                       is_human_source_video=False)
        summary["valid_pick_source"] = bool(source_valid(summary))
        out.mkdir(parents=True, exist_ok=True)
        (out / "source_trace.json").write_text(json.dumps(source.trace, indent=2))
        (out / "source_summary.json").write_text(json.dumps(summary, indent=2))
        return summary, source.trace
    finally:
        env.close()


def _source_route(trace, start):
    """XY from measured source states; never hallucinate a manual Pick path."""
    if not trace:
        return []
    xyz = np.asarray([r["cube"] for r in trace], dtype=float)
    # Raw lifted pick effect: x/y path is copied, then made target-table-supported.
    # Only a small set of temporally ordered raw points is used as push subgoals.
    indices = np.linspace(0, len(xyz) - 1, 7).astype(int)[1:-1]
    route = []
    for i in indices:
        xy = xyz[i, :2].tolist()
        if np.linalg.norm(np.asarray(xy) - np.asarray(start[:2])) > .035:
            if not route or np.linalg.norm(np.asarray(xy) - route[-1]) > .035:
                route.append(xy)
    return route


def run_target(official, seed, method, via, start_ref, goal_ref,
               out, max_steps):
    env, start, goal = make_env(official, seed)
    if not (np.allclose(start, start_ref, atol=.002) and
            np.allclose(goal, goal_ref, atol=.002)):
        env.close()
        raise RuntimeError("Paired initial state/goal mismatch; refuse comparison")
    controller = RoutePush(env, via)
    actual = False
    try:
        for _ in range(max_steps):
            action = controller.act()
            _, _, terminated, truncated, info = env.step(action)
            actual = success_flag(info)
            if actual or bool(np1(terminated)[0]) or bool(np1(truncated)[0]):
                break
        final = np1(env.unwrapped.obj.pose.p)[:3]
        result = dict(type="CUSTOM_ADDITIONAL_REAL_MANISKILL_PHYSICS",
                      is_learned_policy=False,
                      baseline=method, seed=seed, success=actual,
                      steps=len(controller.trace),
                      intermediate_reached=len(controller.visited),
                      requested_intermediate=len(via),
                      final_goal_error_m=float(np.linalg.norm(final[:2] - goal[:2])),
                      source_and_target_share_task_goal=True,
                      target_gripper_forced_open=True,
                      privileged_object_state_used=True,
                      warning="Privileged-state scripted negative control, not LfO policy.")
        out.mkdir(parents=True, exist_ok=True)
        (out / "physical_result.json").write_text(json.dumps(result, indent=2))
        (out / "physical_trace.json").write_text(json.dumps(
            controller.trace, indent=2))
        return result
    finally:
        env.close()


def run(official, seed, out, max_steps):
    out = Path(out)
    source, trace = run_source(official, seed, out / "source_pick", max_steps)
    report = dict(type="CUSTOM_ADDITIONAL_PICK_TO_PUSH_DEV_PHYSICS",
                  status="SOURCE_INVALID_NO_TARGET_COMPARISON",
                  source=source, targets=[], paper_ready=False)
    if not source["valid_pick_source"]:
        (out / "status.json").write_text(json.dumps(report, indent=2))
        print("CUSTOM_PHYSICS_SOURCE_GATE", json.dumps(report["status"]))
        return report
    start, goal = source["initial_cube"], source["goal"]
    candidates = {
        "goal_only": [],
        "waypoint_oracle": route_via(start),
        "raw_source_xy": _source_route(trace, start),
    }
    # Keep geometry/manual-oracle distinct from future learned effect projection.
    for method in METHODS:
        report["targets"].append(run_target(
            official, seed, method, candidates[method], start, goal,
            out / method, max_steps))
    report["status"] = "DEV_SIM_PHYSICS_COMPLETE_NOT_TRO_RESULTS"
    (out / "status.json").write_text(json.dumps(report, indent=2))
    print("CUSTOM_PHYSICS_DEV", json.dumps({
        "status": report["status"],
        "source_valid": True,
        "targets": {r["baseline"]: r["success"] for r in report["targets"]}}))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", default="third_party/roboreel_models")
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=450)
    p.add_argument("--out", default="outputs/v140_custom_pick_push_dev")
    args = p.parse_args()
    run(args.official_root, args.seed, args.out, args.max_steps)


if __name__ == "__main__":
    main()
