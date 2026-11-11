"""Offline stdlib B01 acquisition guards. No network, training or PhysX."""
import tempfile
import unittest
import json
from pathlib import Path

from scripts.prepare_b01_robot_train import (
    TARGET, BUDGET_MB, selected_train_guard,
    frozen_split, make_local_report, write_once,
)


def fixture():
    demos=[]
    for k in range(64):
        code=f"{k:04d}"
        root=f"robot_trajectories/no_distraction/push_cube/{code}/"
        files=[{"path":root+name,"bytes":1024} for name in
               ("trajectory.h5","base_camera.mp4","hand_camera.mp4")]
        demos.append({"demo":code,"files":files})
    return {"task":"push_cube","demos":demos,"bytes":64*3*1024}


class B01TrainOnlyUnitTests(unittest.TestCase):
    def test_real_train_only_fixed_count_and_no_eval(self):
        plan=fixture()
        ids=selected_train_guard(plan)
        self.assertEqual(len(ids),TARGET)
        self.assertEqual(ids[0],"0000")
        self.assertEqual(ids[-1],"0063")
        plan["demos"][-1]["demo"]="0080"
        with self.assertRaisesRegex(ValueError,"TRAIN/EVAL"):
            selected_train_guard(plan)

    def test_source_path_and_budget_guards(self):
        plan=fixture()
        plan["demos"][0]["files"][0]["path"] = (
            "human_demos/eval/no_distraction/push_cube/0080/front.mp4")
        with self.assertRaisesRegex(ValueError,"incorrect official"):
            selected_train_guard(plan)
        plan=fixture()
        plan["bytes"]=301*1024*1024
        with self.assertRaisesRegex(ValueError,"budget"):
            selected_train_guard(plan)
        plan=fixture()
        plan["demos"]=plan["demos"][:-1]
        with self.assertRaisesRegex(ValueError,"64 complete"):
            selected_train_guard(plan)

    def test_validation_split_is_fixed_disjoint_and_no_test(self):
        ids=selected_train_guard(fixture())
        split=frozen_split(ids)
        self.assertEqual(split["count_train"],51)
        self.assertEqual(split["count_val"],13)
        self.assertEqual(split,frozen_split(ids))
        self.assertFalse(set(split["train_demo_ids"]) &
                         set(split["validation_demo_ids"]))
        self.assertTrue(all(int(i)<80 for i in
            split["train_demo_ids"]+split["validation_demo_ids"]))

    def test_actual_h5_audit_required_before_any_training_claim(self):
        ids=selected_train_guard(fixture())
        manifest={
            "status":"REAL_ROBOT_TRAIN_DATA_ACQUIRED",
            "downloaded":[{"demo":i} for i in ids],
        }
        audits={"status":"REAL_ACTION_DATA_VALIDATED","valid_demos":64,
                "rows":[{"demo":i,"action_shape":[138,8]} for i in ids]}
        record=make_local_report(manifest,audits)
        self.assertEqual(record["total_real_robot_action_steps"],64*138)
        self.assertEqual(record["observed_action_dim"],8)
        self.assertFalse(record["upstream_checkpoint_training_budget_comparable"])
        self.assertTrue(record["no_model_training_performed"])
        self.assertTrue(record["custom_pick_push_m1_result_not_rescued"])
        broken=dict(audits,valid_demos=63)
        with self.assertRaisesRegex(ValueError,"audit failed"):
            make_local_report(manifest,broken)

    def test_dont_replace_existing_real_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"frozen.json"
            write_once(file,{"version":1})
            write_once(file,{"version":1})
            with self.assertRaises(FileExistsError):
                write_once(file,{"version":2})


if __name__=="__main__":
    unittest.main()
