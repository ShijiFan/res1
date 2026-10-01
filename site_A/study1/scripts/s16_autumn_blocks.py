"""Site 1: rebuild per-target, per-block autumn confusion arrays from the saved predictions.

Reason (found 2026-10-01): s7_autumn_eval_v7.py (copy of tgrs_revision/scripts/run_external_autumn_eval.py)
reports as "ci_95" the 2.5/97.5 percentiles of the 36 per-(seed, source, target) cell differences. That is
the spread of cells, not a confidence interval of the mean increment, and it is not the BCa block bootstrap
described in the manuscript. Its saved blocks_confusion.npy pools all four targets, so the arrays are rebuilt
here from runs_ext/*/predictions.parquet, in the (4 targets, n_blocks, 5, 5) layout used everywhere else.
The block bootstrap / BCa itself is computed by site2_20260930/scripts/s2_18_study1_autumn.py:stats so
that both sites use one implementation.
Out: ../ws/runs_autumn/<run>/blocks_confusion.npy, then ../ws/tables/autumn_bca.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "ws"
SRC = WS / "autumn" / "runs_ext"
TR = ["t15", "t37", "t88", "t139"]


def main():
    split = json.loads((WS / "splits" / "split_manifest.json").read_text())["historical_development_split"]
    teb = split["eval_blocks"]
    n = 0
    for d in sorted(SRC.iterdir()):
        p = pd.read_parquet(d / "predictions.parquet")
        bc = np.zeros((4, len(teb), 5, 5), dtype=np.int64)
        for ti, tgt in enumerate(TR):
            q = p[p.target_track == tgt]
            for bi, b in enumerate(teb):
                m = q.block_id == b
                bc[ti, bi] = confusion_matrix(q.y_true[m], q.y_pred[m], labels=[0, 1, 2, 3, 4])
        od = WS / "runs_autumn" / d.name
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / "blocks_confusion.npy", bc)
        n += 1
    print("rebuilt", n, "runs; eval pixels per target", int(bc[0].sum()))
    sys.path.insert(0, str(ROOT.parents[0] / "site2_20260930" / "scripts"))
    import s2_18_study1_autumn as s18
    s18.WS = WS
    s18.stats(int(bc[0].sum()), 9600)
    (WS / "tables" / "autumn.csv").rename(WS / "tables" / "autumn_bca.csv")


if __name__ == "__main__":
    main()
