"""Offline tests for metadata-only original RoboReel SigLIP availability."""
from scripts import audit_official_siglip_dataset as audit


def test_all_twenty_eval_bundles_have_to_be_present():
    paths = [f"video_demonstration_features/without_distraction/push_cube/{i:04d}/front.npy"
             for i in range(80, 100)]
    report=audit.audit(paths)
    assert report["eval_front_bundles_found"]==20
    assert report["safe_to_skip_base_download_if_original_schema_verified"]
    assert report["missing_ids"]==[]
    assert report["network_model_downloaded"] is False


def test_missing_bundle_not_hidden_by_video_or_other_camera():
    paths = ["human_demos/eval/no_distraction/push_cube/0080/front.mp4",
             "video_demonstration_features/without_distraction/push_cube/0080/left.npy",
             "video_demonstration_features/with_distraction/push_cube/0080/front.npy"]
    report=audit.audit(paths)
    assert report["eval_front_bundles_found"]==0
    assert report["missing_ids"][0]=="0080"
    assert len(report["missing_ids"])==20
