"""Offline, physics-free design contract for a counterfactual video-goal push task.

This module does NOT certify human demos, controller reachability or G0/G1.
It defines two mutually exclusive evaluation targets for an identical cube
spawn. Goal labels and target positions must never be included in policy obs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Mapping, Any

GOALS = ("-x", "+x")
SENSITIVE_TOKENS = ("goal", "target", "desired", "reward", "success", "direction", "sign")


@dataclass(frozen=True)
class CounterfactualPushSpec:
    distance_m: float = 0.15
    radius_m: float = 0.05
    cube_half_size_m: float = 0.02
    cube_spawn_xy_bound_m: float = 0.10
    table_xy_bound_m: float = 0.40  # conservative design-space bound, NOT physics reachability

    def __post_init__(self) -> None:
        scalars = (self.distance_m, self.radius_m, self.cube_half_size_m,
                   self.cube_spawn_xy_bound_m, self.table_xy_bound_m)
        if not all(isfinite(v) and v > 0 for v in scalars):
            raise ValueError("All length parameters must be positive finite values")
        if 2 * self.radius_m >= 2 * self.distance_m:
            raise ValueError("Opposite goal regions must be geometrically disjoint")
        if self.cube_spawn_xy_bound_m + self.distance_m + self.radius_m > self.table_xy_bound_m:
            raise ValueError("Both goal zones must remain within the conservative table design bound")
        if self.radius_m < self.cube_half_size_m:
            raise ValueError("Evaluation goal radius must not be narrower than cube half size")

    def targets(self, cube_xyz: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
        if len(cube_xyz) != 3 or not all(isfinite(float(v)) for v in cube_xyz):
            raise ValueError("Expected finite cube (x,y,z) position")
        x, y, _ = map(float, cube_xyz)
        if abs(x) > self.cube_spawn_xy_bound_m or abs(y) > self.cube_spawn_xy_bound_m:
            raise ValueError("Cube spawn falls outside frozen evaluation spawn limits")
        return {"-x": (x - self.distance_m, y, self.cube_half_size_m),
                "+x": (x + self.distance_m, y, self.cube_half_size_m)}

    def goal_reached(self, cube_xyz: tuple[float, float, float],
                     target: tuple[float, float, float]) -> bool:
        if len(cube_xyz) != 3 or len(target) != 3:
            raise ValueError("Need valid cube and target XYZ")
        if not all(isfinite(float(z)) for z in tuple(cube_xyz) + tuple(target)):
            raise ValueError("Positions must be finite")
        x, y, z = map(float, cube_xyz)
        gx, gy, _ = map(float, target)
        return hypot(x-gx, y-gy) < self.radius_m and z < (self.cube_half_size_m + 0.005)

    def paired_design(self, cube_xyz: tuple[float, float, float], reset_seed: int) -> dict[str, Any]:
        if type(reset_seed) is not int or not 0 <= reset_seed <= 2**32-1:
            raise ValueError("Integer reset seed must be within 32-bit range")
        opposite = self.targets(cube_xyz)
        return {"reset_seed": reset_seed, "same_initial_cube_xyz": list(map(float,cube_xyz)),
                "goal_radius_m": self.radius_m,
                "goal_dx_m": {label: opposite[label][0]-float(cube_xyz[0]) for label in GOALS},
                "targets_evaluator_only": {label: list(opposite[label]) for label in GOALS},
                "targets_not_policy_input": True, "world_frame_not_camera_left_right": True,
                "physics_reachability_established": False,
                "human_video_goal_pair_established": False}


def assert_no_goal_leak(policy_observation: Mapping[str, Any]) -> None:
    """Structural anti-leak gate for future policy adapter; not inference code."""
    if not isinstance(policy_observation, Mapping):
        raise ValueError("Policy observation must be a mapping")
    def walk(prefix: str, value: Any):
        for field in prefix.replace("/", "_").lower().split("_"):
            if any(k in field for k in SENSITIVE_TOKENS):
                raise ValueError("GOAL_INFORMATION_LEAK_IN_POLICY_OBSERVATION: " + prefix)
        if isinstance(value, Mapping):
            for key, sub in value.items():
                walk(prefix + "/" + str(key), sub)
    for key, value in policy_observation.items():
        walk(str(key), value)
