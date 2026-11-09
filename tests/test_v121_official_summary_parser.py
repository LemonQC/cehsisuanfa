"""Actual upstream run_eval_protocol summary shape, no made-up physics results."""
from agcofl.vid2robot_official_smoke import extract_official_single_trial


def test_official_per_trial_dict_yields_single_real_result():
    row=dict(task="push_cube",demo_id=80,success=False)
    summary=dict(per_trial={"demo_distract=False|env_distract=False":[row]},
                 table={"demo_distract=False|env_distract=False":{
                     "success":0,"total":1,"rate":0.0}})
    assert extract_official_single_trial(summary,"push_cube",80)==row


def test_official_per_trial_true_is_not_confused_with_missing_trial():
    row=dict(task="push_cube",demo_id=80,success=True)
    summary=dict(per_trial={"demo_distract=False|env_distract=False":[row]})
    assert extract_official_single_trial(summary,"push_cube",80)["success"] is True
    assert extract_official_single_trial(summary,"push_cube",81) is None


def test_no_physics_trial_or_many_trials_never_returns_success():
    assert extract_official_single_trial({}, "push_cube",80) is None
    assert extract_official_single_trial({"per_trial":{}}, "push_cube",80) is None
    x=dict(task="push_cube",demo_id=80,success=True)
    assert extract_official_single_trial({"per_trial":{"one":[x],"two":[x]}},
                                         "push_cube",80) is None
    assert extract_official_single_trial({"per_trial":{"one":[{
       "task":"push_cube","demo_id":80,"success":"True"}]}},
                                         "push_cube",80) is None
