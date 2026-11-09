"""No-policy/no-GPU regression of official live observability instrumentation.

These fixtures do NOT generate physical results; they verify:
- the original author methods receive unchanged arguments, once per episode;
- only a returned ORIGINAL rollout result counts as completed;
- a physical exception cannot turn into a success entry;
- progress files are user-observable while an actual seed pass is running.
"""
import json
import pytest

from agcofl.official_live_progress import install_official_progress


class DummyOfficialRunner:
    def __init__(self):
        self.calls=[]
    def load_demo_static(self, demo_distract, task, demo_id):
        self.calls.append(("load",demo_distract,task,demo_id))
        return {"demo":demo_id}
    def rollout(self, env, obs, demo_static, max_steps, chunk_size, video_frames):
        self.calls.append(("rollout",env,obs,demo_static,max_steps,
                           chunk_size,video_frames))
        if env=="crash":
            raise RuntimeError("physical simulator interrupted")
        return env=="success"


def test_original_actions_and_returns_are_preserved(tmp_path,capsys):
    r=DummyOfficialRunner()
    path=install_official_progress(r,tmp_path,seed_offset=100000,total=20)
    a=r.load_demo_static(False,"push_cube",80)
    assert a=={"demo":80}
    assert r.rollout("success","rgb",a,800,1,None) is True
    b=r.load_demo_static(False,"push_cube",81)
    assert b=={"demo":81}
    assert r.rollout("failure","rgb",b,800,1,None) is False

    assert r.calls==[
        ("load",False,"push_cube",80),
        ("rollout","success","rgb",a,800,1,None),
        ("load",False,"push_cube",81),
        ("rollout","failure","rgb",b,800,1,None),
    ]
    state=json.loads(path.read_text())
    assert state["seed_offset_base"]==100000
    assert state["completed"]==2
    assert state["total_expected"]==20
    assert state["successes_so_far"]==1
    assert state["percentage"]==10.0
    assert state["active_demo_id"] is None
    assert state["last_completed"]["demo_id"]==81
    assert state["last_completed"]["success"] is False
    assert state["status"]=="RUNNING"
    events=[json.loads(x) for x in (tmp_path/"progress.jsonl").read_text().splitlines()]
    assert [x["event"] for x in events]==[
        "MONITOR_START","START","DONE","START","DONE"]
    log=capsys.readouterr().out
    assert "OFFICIAL_V130_TRIAL_DONE" in log
    assert '"index": 2' in log


def test_error_does_not_manufacture_success_or_completed_count(tmp_path):
    r=DummyOfficialRunner()
    path=install_official_progress(r,tmp_path,seed_offset=0,total=20)
    static=r.load_demo_static(False,"push_cube",80)
    with pytest.raises(RuntimeError,match="simulator interrupted"):
        r.rollout("crash","rgb",static,800,1,None)
    state=json.loads(path.read_text())
    assert state["completed"]==0
    assert state["successes_so_far"]==0
    assert state["status"]=="ERROR"
    assert not (tmp_path/"summary.json").exists()


def test_completed_trial_marker_never_claims_official_summary(tmp_path):
    r=DummyOfficialRunner()
    path=install_official_progress(r,tmp_path,seed_offset=200000,total=1)
    s=r.load_demo_static(False,"push_cube",99)
    r.rollout("success","rgb",s,800,1,None)
    state=json.loads(path.read_text())
    assert state["status"]=="ALL_TRIALS_RETURNED_AWAIT_OFFICIAL_SUMMARY"
    assert state["percentage"]==100.0
    assert state["completed"]==1
    assert "summary.json" in state["authoritative_results_path"]
    assert not (tmp_path/"summary.json").exists()
