"""Plot paired Full / Anchor / End metrics from offline experiment CSVs."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def make_plots(folder):
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    folder=Path(folder)
    with (folder/"robot_test_summary.csv").open() as f:
        summary=list(csv.DictReader(f))
    with (folder/"training_history.csv").open() as f:
        history=list(csv.DictReader(f))
    if not summary or not history:
        raise ValueError("Run partial_flow_study train first")
    conditions=sorted({x["condition"] for x in summary})
    modes=("full","anchor","end")
    x=np.arange(len(conditions))
    fig,ax=plt.subplots(figsize=(8,4))
    for i,mode in enumerate(modes):
        y=[]
        for c in conditions:
            rows=[r for r in summary if r["condition"]==c and r["mode"]==mode]
            y.append(float(rows[0]["mean_full_iou"]) if rows else 0.)
        ax.bar(x+(i-1)*.24,y,width=.22,label=mode)
    ax.set_ylim(0,1.0)
    ax.set_xticks(x,conditions)
    ax.set_ylabel("Offline robot-test full-trajectory mask IoU")
    ax.set_title("Offline diagnostic only — NOT robot task success")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/"offline_comparison.png",dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    for mode in modes:
        rows=[r for r in history if r["mode"]==mode]
        ax.plot([int(r["epoch"]) for r in rows],
                [float(r["train_loss"]) for r in rows],label=mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training objective (different label sets)")
    ax.set_title("Diagnostic learning curves — not directly comparable objectives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/"training_curves.png",dpi=160)
    plt.close(fig)
    print("PLOTS_WRITTEN",folder/"offline_comparison.png",
          folder/"training_curves.png",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    a=p.parse_args()
    make_plots(a.out)


if __name__=="__main__":main()
