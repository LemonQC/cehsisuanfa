"""Offline-only checks for official Vid2Robot asset selection and budget.

Test data are metadata-shaped mock objects, NEVER experimental trajectories.
"""
import pytest

from agcofl.official_vid2robot_assets import model_plan,eval_plan,MODEL_PREFIX
from agcofl.official_robot_data import BASE,plan

class RepoFile:
    def __init__(self,path,size):
        self.path=path
        self.size=size

class MockApi:
    def __init__(self,records):
        self.records=records
    def list_repo_tree(self,repo_type=None,path_in_repo=None,**kwargs):
        assert repo_type in ("model","dataset")
        assert path_in_repo
        return (x for x in self.records if x.path.startswith(path_in_repo+"/"))


def test_dataset_uses_published_bucket_not_source_code_alias():
    assert BASE=="robot_trajectories/no_distraction"
    files=[{"path":f"{BASE}/push_cube/0000/{name}","bytes":1024} for name in
        ("trajectory.h5","base_camera.mp4","hand_camera.mp4")]
    assert len(plan(files,max_demos=2,max_mb=20)["demos"])==1


def test_metadata_only_official_weights_and_bound():
    fake=MockApi([RepoFile("nd/vid2robot/ckpt.pt",3*1024**3),
                  RepoFile("nd/vid2robot/train_config.json",5000)])
    d=model_plan(fake,max_gb=8)
    assert d["status"]=="OFFICIAL_CHECKPOINT_DOWNLOAD_APPROVED"
    assert len(d["files"])==2
    d=model_plan(fake,max_gb=1)
    assert d["status"]=="CHECKPOINT_EXCEEDS_DOWNLOAD_BUDGET"


def test_metadata_only_actual_held_out_front_videos():
    files=[RepoFile(f"human_demos/eval/no_distraction/push_cube/{i:04d}/front.mp4",2_000_000)
           for i in range(80,100)]
    files += [RepoFile("human_demos/train/no_distraction/push_cube/0000/front.mp4",2_000_000),
              RepoFile("human_demos/eval/no_distraction/push_cube/0079/front.mp4",2_000_000)]
    d=eval_plan(MockApi(files),task="push_cube",max_mb=250)
    assert d["status"]=="OFFICIAL_EVAL_VIDEOS_DOWNLOAD_APPROVED"
    assert len(d["files"])==20
    assert all(int(x["path"].split("/")[-2])>=80 for x in d["files"])


def test_partial_eval_data_must_not_be_called_official():
    d=eval_plan(MockApi([RepoFile("human_demos/eval/no_distraction/push_cube/0080/front.mp4",100)]))
    assert d["status"]=="EVAL_SUBSET_INCOMPLETE"
    assert d["found"]==1


def test_checkpoint_without_real_train_config_is_blocked():
    fake=MockApi([RepoFile("nd/vid2robot/ckpt.pt",1000)])
    assert model_plan(fake)["status"]=="OFFICIAL_CHECKPOINT_FILES_MISSING"
