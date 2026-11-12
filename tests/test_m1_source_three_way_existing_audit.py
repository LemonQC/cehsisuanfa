"""CPU-only frozen three physical source traces, no simulator access."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import m1_source_three_way_existing_audit as mod


def case(root,label,goal,steps,grasp,close_step):
    root.mkdir(parents=True)
    trace=[]
    for t in range(steps):
        cube=[0.,0.,.02 + (.018 if grasp and t>=close_step+2 else 0.)]
        tcp=[-.010,.0,.12 if label=="side" else (.02 if t>=close_step else .12)]
        trace.append({
           "t":t,"tcp":tcp,"task_cube":cube,"blocker":[.1,0.,.029],
           "gripper_action":-1. if t>=close_step else 1.,
           "is_grasping_task":bool(grasp and t>=close_step+1),
        })
    if label=="b3":
        status={"kind":"source_pick_to_same_goal",
                "mode":"blocked_source","scene":mod.SCENE,
                "seed":mod.SEED,"steps":steps,
                "gate":{"valid":False},
                "state_before":[0,0,.02],"goal":[goal,0,.001],
                "planner_phases":[{"name":"grasp_pose","planner_success":True}]}
    elif label=="side":
        status={"kind":"source_side_grasp","version":
                "m1_current_task_official_executor_swap_v1",
                "original_geometry":True,"seed":mod.SEED,
                "steps":steps,"success":False,
                "initial_cube":[0,0,.02],"goal":[goal,0,.001],
                "planner_phases":[{"stage":"physical_grasp_check",
                                   "actual_is_grasping":False}]}
    else:
        status={"kind":"source_pick_to_same_goal",
                "steps":steps,"gate":{"valid":True},
                "state_before":[0,0,.02],"goal":[goal,0,.001]}
    (root/"status.json").write_text(json.dumps(status))
    (root/"physical_trace.json").write_text(json.dumps(trace))


class PriorSourceThreeWay(unittest.TestCase):
    def setup_three(self,root):
        b3,side,old=(root/q for q in ("b3","side","old"))
        case(b3,"b3",.21,20,False,6)
        case(side,"side",.21,12,False,6)
        case(old,"old",.15,25,True,5)
        return b3,side,old

    def test_all_three_real_contact_types_and_goal_scope(self):
        with tempfile.TemporaryDirectory() as d:
            roots=self.setup_three(Path(d))
            r=mod.analyse(*roots)
            self.assertEqual(r["actual_physx_new_runs"],0)
            self.assertEqual(r["same_source_target_goal_x_m"],.21)
            self.assertEqual(r["old_reference_different_goal_x_m"],.15)
            self.assertEqual(r["cases"]["b3_side_grasp"]["actual_grasping_step_count"],0)
            self.assertGreater(r["cases"]["b2_success_reference"]["actual_grasping_step_count"],0)
            self.assertEqual(len(r["inputs_sha256"]["original_b3"]["trace_sha256"]),64)

    def test_b3_near_cube_but_no_grasp_is_not_credited_success(self):
        with tempfile.TemporaryDirectory() as d:
            roots=self.setup_three(Path(d))
            r=mod.analyse(*roots)
            self.assertTrue(any("physical finger/tool orientation"
                                in x for x in r["findings"]))
            self.assertFalse(r["source_pair_success_claimed"])

    def test_rejects_wrong_goal_or_fake_success(self):
        with tempfile.TemporaryDirectory() as d:
            b3,side,old=self.setup_three(Path(d))
            p=side/"status.json"
            s=json.loads(p.read_text());s["goal"][0]=.15
            p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,"share real target goal"):
                mod.analyse(b3,side,old)
            s["goal"][0]=.21;s["success"]=True
            p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,"genuine failed"):
                mod.analyse(b3,side,old)

    def test_rejects_missing_grasp_flags_or_step_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            b3,side,old=self.setup_three(Path(d))
            p=b3/"physical_trace.json"
            data=json.loads(p.read_text())
            data[4]["t"]=123
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"invalid actual step"):
                mod.analyse(b3,side,old)


if __name__=="__main__":
    unittest.main()
