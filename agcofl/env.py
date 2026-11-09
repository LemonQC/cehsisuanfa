"""Two-route, two-stage tabletop cube task built on ManiSkill 3 PushCubeEnv.

Physical barriers are static. Route side is intentionally NOT included in policy
observation: only the future reference mask communicates that preference.
"""
from __future__ import annotations
import numpy as np
import torch
import sapien
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

START = np.array([-0.145, 0.0], dtype=np.float32)
GOAL = np.array([0.145, 0.0], dtype=np.float32)
CHECK_Y = 0.155
GOAL_TOL = 0.055
CHECK_TOL = 0.065


def route_waypoints(side: int) -> np.ndarray:
    if int(side) not in (-1, 1):
        raise ValueError('side must be -1 or 1')
    return np.array([START, [-0.105, side * CHECK_Y], [0.0, side * CHECK_Y],
                     [0.105, side * CHECK_Y], GOAL], dtype=np.float32)


@register_env('AGCOFL-TwoRoute-v1', max_episode_steps=550)
class AGCOFLTwoRoute(PushCubeEnv):
    goal_radius = GOAL_TOL

    def __init__(self, *args, route_side=1, **kwargs):
        self.route_side = int(route_side)
        if self.route_side not in [-1, 1]:
            raise ValueError('route_side must be ±1')
        super().__init__(*args, robot_uids='panda', robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_pose = sapien_utils.look_at(eye=[0.05, -0.02, 0.75], target=[0.0, 0.0, 0.0])
        return [CameraConfig('base_camera', pose=camera_pose, width=128, height=128,
                             fov=np.pi/2, near=0.01, far=10.0)]

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        # Raised barrier between the two valid tabletop channels.
        builder.add_box_collision(half_size=[0.050, 0.077, 0.045])
        builder.add_box_visual(half_size=[0.050, 0.077, 0.045],
                               material=sapien.render.RenderMaterial(base_color=[0.45, 0.45, 0.5, 1]))
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.045]))
        self.barrier = builder.build_static(name='route_barrier')
        # Small colored visual checkpoint marker (no collision).
        marker = self.scene.create_actor_builder()
        marker.add_box_visual(half_size=[0.035, 0.035, 0.0005],
                              material=sapien.render.RenderMaterial(base_color=[0.1, 0.8, 0.2, 0.65]))
        marker.set_initial_pose(sapien.Pose(p=[0, CHECK_Y * self.route_side, 0.001]))
        self.check_marker = marker.build_kinematic(name='inspection_zone')

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Parent resets robot and table; override cube and goal without altering base mechanics.
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        with torch.device(self.device):
            start = torch.tensor([float(START[0]), float(START[1]), float(self.cube_half_size)]).repeat(b, 1)
            goal = torch.tensor([float(GOAL[0]), float(GOAL[1]), 0.001]).repeat(b, 1)
            self.obj.set_pose(Pose.create_from_pq(p=start, q=[1, 0, 0, 0]))
            self.goal_region.set_pose(Pose.create_from_pq(p=goal, q=[1, 0, 0, 0]))
            marker_xyz = torch.tensor([0.0, self.route_side * CHECK_Y, .001]).repeat(b, 1)
            self.check_marker.set_pose(Pose.create_from_pq(p=marker_xyz, q=[1,0,0,0]))
            self._check_visited = torch.zeros((self.num_envs,), dtype=torch.bool)

    def evaluate(self):
        xy = self.obj.pose.p[..., :2]
        check_xy = torch.tensor([[0.0, self.route_side * CHECK_Y]], device=xy.device)
        visited = torch.linalg.norm(xy - check_xy, dim=-1) < CHECK_TOL
        self._check_visited |= visited
        at_goal = torch.linalg.norm(xy - self.goal_region.pose.p[..., :2], dim=-1) < GOAL_TOL
        on_table = torch.abs(self.obj.pose.p[..., 2] - self.cube_half_size) < .012
        return {'success': self._check_visited & at_goal & on_table,
                'check_visited': self._check_visited.clone(), 'at_goal': at_goal,
                'on_table': on_table}

    def _get_obs_extra(self, info):
        # DO NOT reveal route_side, checkpoint marker or hidden check state in policy observation.
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose,
                    goal_pos=self.goal_region.pose.p,
                    obj_pose=self.obj.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        xy = self.obj.pose.p[..., :2]
        check = torch.tensor([[0.0, self.route_side * CHECK_Y]], device=xy.device)
        dcheck = torch.linalg.norm(xy-check, dim=-1)
        dgoal = torch.linalg.norm(xy-self.goal_region.pose.p[..., :2], dim=-1)
        return (1 - torch.tanh(5 * torch.where(self._check_visited, dgoal, dcheck))) + info['success'].float() * 3

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)/4.
