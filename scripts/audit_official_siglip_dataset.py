#!/usr/bin/env python3
"""Metadata-only search for official RoboReel held-out SigLIP .npy bundles.

DOES NOT download any model, video, feature, or authenticate with GCS.
This only checks whether the public RoboReel HF DATASET actually publishes
feature bundles for EVAL PushCube IDs 0080..0099. Absence is evidence
of no public precomputed bundles at the queried revision, not of network
inaccessibility to GCS.
"""
from __future__ import annotations
import argparse
import json
from urllib.parse import urlsplit

from huggingface_hub import HfApi

DATASET="LogosRoboticsLab/roboreel"
IDS=tuple(range(80,100))


def audit(paths: list[str]):
    by_id={}
    for demo_id in IDS:
        suffix=f"push_cube/{demo_id:04d}/front.npy"
        by_id[f"{demo_id:04d}"] = sorted(
            p for p in paths
            if "video_demonstration_features/" in p
            and p.endswith(suffix)
            and ("/without_distraction/" in p or "/no_distraction/" in p)
        )
    possible = sorted(
        p for p in paths
        if p.startswith("video_demonstration_features/")
        or "/video_demonstration_features/" in p
        or "token_emb_" in p
    )
    missing=[key for key,values in by_id.items() if not values]
    return {
        "type":"ROBOREEL_OFFICIAL_DATASET_SIGLIP_METADATA_ONLY",
        "repo_id":DATASET,
        "eval_front_bundles_found":len(IDS)-len(missing),
        "eval_front_bundles_required":len(IDS),
        "missing_ids":missing,
        "matches":{k:v for k,v in by_id.items() if v},
        "derived_feature_file_count":len(possible),
        "derived_feature_file_examples":possible[:12],
        "safe_to_skip_base_download_if_original_schema_verified":not missing,
        "network_model_downloaded":False,
        "physics_executed":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint",default="https://hf-mirror.com")
    p.add_argument("--revision",default="main")
    args=p.parse_args(argv)
    u=urlsplit(args.endpoint)
    if u.scheme!="https" or not u.hostname or u.username or u.password:
        p.error("metadata endpoint must use public HTTPS without embedded credentials")
    api=HfApi(endpoint=args.endpoint.rstrip("/"),token=False)
    info=api.dataset_info(DATASET,revision=args.revision,files_metadata=False,token=False)
    report=audit([s.rfilename for s in info.siblings])
    report["revision"]=info.sha
    print("OFFICIAL_SIGLIP_DATASET_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
    if not report["safe_to_skip_base_download_if_original_schema_verified"]:
        print("NO_COMPLETE_PUBLIC_ORIGINAL_EVAL_SIGLIP_BUNDLES: source repo "
              "README requires computing derived artifacts. Do not invent .npy files.",flush=True)
        return 2
    print("PUBLIC_FEATURES_POTENTIALLY_AVAILABLE: verify source and .npy schema "
          "before downloading ONLY those files.",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
