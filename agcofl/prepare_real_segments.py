"""Convert real robot physics episodes or calibrated human object tracks to mask clips.

Input is JSONL with explicit layout IDs and known task goal coordinates.
No synthetic source is fabricated by this script. A trajectory extracted from
a real human video is still a *coordinate-mask projection*, not an RGB image.
For native human segmentation masks, use the direct manifest format in docs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .partial_flow_study import masks_from_points, context_of


def convert(index, out):
    index=Path(index).resolve();root=index.parent
    out=Path(out);(out/"masks").mkdir(parents=True,exist_ok=True)
    entries=[json.loads(x) for x in index.read_text().splitlines() if x.strip()]
    manifest=[];skipped=[]
    for row in entries:
        for field in ("id","layout_id","source","condition","path",
                      "check_goal","final_goal","scene"):
            if field not in row:raise ValueError(f"{row.get('id','?')} missing {field}")
        if row["source"] not in ("human","robot"):
            raise ValueError("source must be human or robot; no human_proxy in real converter")
        src=(root/row["path"]).resolve()
        if not src.is_relative_to(root.resolve()):
            raise ValueError(f"path must be under index folder: {src}")
        if row["source"]=="robot":
            with np.load(src) as d:
                if "success" not in d or not bool(d["success"]):
                    skipped.append({"id":row["id"],"reason":"failed robot episode"})
                    continue
                if not {"cube_before","cube_after","visited"}.issubset(d.files):
                    raise ValueError(f"{src}: missing recorded physical trajectory")
                after=np.asarray(d["cube_after"],np.float32)[:,:2]
                xy=np.vstack([np.asarray(d["cube_before"],np.float32)[0,:2],after])
                visits=np.asarray(d["visited"],bool).reshape(-1)
                hit=np.flatnonzero(visits)
                if len(hit)==0:
                    skipped.append({"id":row["id"],"reason":"no real checkpoint event"})
                    continue
                boundary=int(hit[0])+1
        else:
            if "boundary_index" not in row:
                raise ValueError(f"human {row['id']} needs annotated boundary_index")
            xy=np.asarray(np.load(src),np.float32)
            if xy.ndim!=2 or xy.shape[1]!=2:
                raise ValueError(f"human {src} must contain calibrated world XY [N,2]")
            boundary=int(row["boundary_index"])
        if len(xy)<12 or not 5<=boundary<=len(xy)-6:
            skipped.append({"id":row["id"],"reason":"segment too short/bad checkpoint boundary"})
            continue
        for phase,points,goal in (
          (0,xy[:boundary+1],row["check_goal"]),
          (1,xy[boundary:],row["final_goal"])):
            mask=masks_from_points(points)
            uid=f"{row['id']}_P{phase}"
            rel=f"masks/{uid}.npy"
            np.save(out/rel,mask)
            manifest.append(dict(
                id=uid,layout_id=str(row["layout_id"]),
                condition=row["condition"],phase=phase,source=row["source"],
                mask_path=rel,
                context=context_of(points[0],goal,phase,row["scene"]),
                source_recording=str(row["path"]),
                caveat="Calibrated world-XY silhouette proxy, not video RGB segmentation"
            ))
    path=out/"manifest.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in manifest))
    report=dict(index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                inputs=len(entries),clips=len(manifest),skipped=skipped,
                warning="Physical robot actions only count when stored success=True. "
                        "Human world-XY must be calibrated from REAL human recordings. "
                        "Use independent layouts for held-out generalization.")
    (out/"preparation_report.json").write_text(json.dumps(report,indent=2))
    print("REAL_PREPARED",json.dumps(report),flush=True)
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description="Convert human/robot tracks into paired masks")
    p.add_argument("--index",required=True)
    p.add_argument("--out",default="datasets/real_mask_segments")
    args=p.parse_args(argv)
    convert(args.index,args.out)


if __name__=="__main__":main()
