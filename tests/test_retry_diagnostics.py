"""Unit tests for retry state machine; does NOT claim physical ManiSkill success."""
import numpy as np
from agcofl import teacher


class _Pose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float32)


class _Robot:
    def get_qpos(self):
        return np.zeros((9,), dtype=np.float32)

    def get_qvel(self):
        return np.zeros((9,), dtype=np.float32)


class _Agent:
    def __init__(self):
        self.robot = _Robot()
        self.tcp = type("TCP", (), {"pose": _Pose([-.20, 0., .15])})()


class _Base:
    def __init__(self):
        self.agent = _Agent()
        self.obj = type("Obj", (), {"pose": _Pose([-.145, 0., .02])})()
        self.goal_region = type("Goal", (), {"pose": _Pose([.145, 0., 0.])})()

    def evaluate(self):
        return {"success": np.asarray([True]), "check_visited": np.asarray([True])}


class _Env:
    def __init__(self):
        self.unwrapped = _Base()
        self.action_space = type("ActionSpace", (), {"shape": (4,)})()

    def reset(self, seed):
        self.unwrapped.obj.pose.p[:] = [-.145, 0., .02]
        return {}, {}

    def step(self, action):
        return {}, 0., False, False, {"check_visited": np.asarray([False])}


def test_teacher_recontacts_using_measured_cube(monkeypatch):
    env = _Env()
    counts = {}

    def fake_leg(rec, point, **kwargs):
        key = tuple(np.asarray(point).round(3))
        counts[key] = counts.get(key, 0) + 1
        old = rec.unwrapped.obj.pose.p[:2].copy()
        if counts[key] == 1:
            rec.unwrapped.obj.pose.p[:2] = .5 * (old + point)
            return False
        rec.unwrapped.obj.pose.p[:2] = point
        return True

    monkeypatch.setattr(teacher, "_push_leg", fake_leg)
    result = teacher.scripted_rollout(env, side=-1, seed=100, max_retries=1)
    assert result["success"], result["error"]
    assert len(result["push_attempts"]) == 6
    assert [a["reached"] for a in result["push_attempts"]] == [False, True] * 3
    assert all(a["progress_m"] > 0 for a in result["push_attempts"])
    assert all(counts[key] == 2 for key in counts)


def test_teacher_persists_failed_retries(monkeypatch, tmp_path):
    env = _Env()

    def failed_leg(rec, point, **kwargs):
        rec.unwrapped.obj.pose.p[:2] += np.asarray([0., .01])
        return False

    monkeypatch.setattr(teacher, "_push_leg", failed_leg)
    result = teacher.scripted_rollout(env, side=1, seed=101, max_retries=2)
    assert not result["success"]
    assert "after 3 measured attempts" in result["error"]
    assert len(result["push_attempts"]) == 3
    path = tmp_path / "failed.npz"
    teacher.save_episode(result, path)
    assert path.with_suffix(".json").exists()
    import json
    meta = json.loads(path.with_suffix(".json").read_text())
    assert len(meta["push_attempts"]) == 3


def test_straight_stroke_merging_preserves_checkpoint_and_flow_route():
    from agcofl.representation import contact_waypoints, route_points, future_masks
    for side in (-1, 1):
        route = route_points(side)
        strokes = contact_waypoints(side)
        assert route.shape == (5, 2)
        assert strokes.shape == (3, 2)
        # The middle of the straight line is no longer a forced contact break.
        assert not np.any(np.all(np.isclose(strokes, route[2]), axis=-1))
        # Both actual corners and the terminal goal are unchanged.
        assert np.allclose(strokes, route[[1, 3, 4]])
        # This new controller scheduling must NOT alter side-specific masks.
        assert future_masks(route[0], side).shape == (4, 32, 32)
        # Push from corner 1 to corner 2 crosses the original checkpoint.
        a, b, checkpoint = strokes[0], strokes[1], route[2]
        d = b - a
        t = np.dot(checkpoint-a, d) / np.dot(d, d)
        assert 0 < t < 1
        assert np.linalg.norm(a + t*d-checkpoint) < 1e-5


def test_final_turn_tries_distinct_contact_offsets(monkeypatch):
    env = _Env()
    calls = []
    from agcofl.representation import contact_waypoints
    final = contact_waypoints(-1)[-1]

    def staged_leg(rec, point, *, tolerance, contact_gap):
        calls.append((np.asarray(point).copy(), float(tolerance), float(contact_gap)))
        if np.linalg.norm(np.asarray(point) - final) < 1e-6:
            if sum(np.linalg.norm(c[0]-final)<1e-6 for c in calls) < 3:
                return False
        rec.unwrapped.obj.pose.p[:2] = point
        return True

    monkeypatch.setattr(teacher, "_push_leg", staged_leg)
    result = teacher.scripted_rollout(env, side=-1, seed=100, max_retries=2)
    assert result["success"]
    assert len(calls) == 5
    assert [round(x[2],3) for x in calls[-3:]] == [0.060,0.045,0.075]
    assert all(x[1] == 0.030 for x in calls[:2])
    assert all(x[1] == 0.045 for x in calls[-3:])


def test_aborted_servo_can_return_false_without_throwing(monkeypatch):
    env = _Env()
    rec = teacher.RecorderEnv(env, -1)
    # Simulate all TCP positioning as feasible, except the actual push.
    called = []

    def fake_servo(recorder, target, **kwargs):
        called.append(kwargs)
        return kwargs.get("stop_when") is None

    monkeypatch.setattr(teacher, "_servo_tcp", fake_servo)
    ok = teacher._push_leg(
        rec, [-.105,-.155], tolerance=0.030, contact_gap=0.060)
    assert ok is False
    assert "tcp_target_error_m" in rec.last_push_telemetry
    assert "cube_push_displacement_m" in rec.last_push_telemetry
    assert any(x.get("abort_when") is not None for x in called)
