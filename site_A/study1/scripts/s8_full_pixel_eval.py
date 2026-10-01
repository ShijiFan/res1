"""S2: evaluate the v7 spring models on ALL consensus (C2) pixels of the 12 evaluation blocks.

Training is unchanged (models from ../ws/runs_R1). The published protocol samples 800 pixels
per evaluation block (165 built-up pixels in total); this script predicts every C2 pixel in
those blocks instead. The spatial unit for inference is still the block, so the output
blocks_confusion.npy files feed the same block bootstrap (s5_t1_v7.py with RUNS_NEW
pointed at ../ws/runs_full).
Only pixels at >= 400 m from every training candidate are evaluated (training buffer is
defined against the evaluation region, so this holds by construction; it is re-checked).
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import load
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_run_R1_v7 as s4  # noqa: E402  (paths, cube, R1 module)

r1 = s4.r1
WS = s4.WS
OUT = WS / "runs_full"
REPS = sys.argv[1:] or ["I28", "I28_G29"]


def main():
    sv = r1.SAR_V2
    c7 = np.load(s4.CUBE)
    v4 = np.load(sv / "data" / "flevoland_datacube_v4_multibaseline.npz")
    cm = np.load(sv / "labels" / "consensus_mask.npz")
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om = np.load(s4.SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy")
    vi = np.where(mmu["mmu_valid"])
    omv = om[vi]
    y_wc = v4["y"][vi].astype(int)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    grid_y, grid_x = v4["grid_y"][vi], v4["grid_x"][vi]
    bids = (grid_y // 125) * 100 + (grid_x // 125)
    coords = np.column_stack([grid_y, grid_x]).astype(float) * 40.0
    split = json.loads((WS / "splits" / "split_manifest.json").read_text())["historical_development_split"]
    teb, trb = split["eval_blocks"], split["train_blocks"]

    ev = np.where(np.isin(bids, teb) & c_mask)[0]
    tr_cand = np.where(np.isin(bids, trb) & c_mask)[0]
    dmin = cKDTree(coords[tr_cand]).query(coords[ev])[0]
    y_ev, b_ev = y_wc[ev], bids[ev]
    summary = {"n_eval_pixels": int(ev.size), "per_class": np.bincount(y_ev, minlength=5).tolist(),
               "min_dist_to_any_train_candidate_m": float(dmin.min())}
    print(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_eval_summary.json").write_text(json.dumps(summary, indent=2))

    feats, _ = r1.extract_features(c7, vi)
    for learner in ("RF", "SVC"):
        for rep in REPS:
            for src in r1.TR:
                for seed in r1.SEEDS:
                    name = f"{learner}_{rep}_{src}_s{seed}"
                    rd = WS / "runs_R1" / name
                    clf = load(rd / "model.joblib")
                    sc = load(rd / "scaler.joblib") if (rd / "scaler.joblib").exists() else None
                    bc = np.zeros((4, len(teb), 5, 5), dtype=np.int64)
                    for ti, tgt in enumerate(r1.TARGET_ORDER):
                        X = feats[rep][tgt][ev]
                        p = clf.predict(sc.transform(X) if sc is not None else X)
                        for bi, b in enumerate(teb):
                            m = b_ev == b
                            bc[ti, bi] = confusion_matrix(y_ev[m], p[m], labels=[0, 1, 2, 3, 4])
                    od = OUT / name
                    od.mkdir(exist_ok=True)
                    np.save(od / "blocks_confusion.npy", bc)
                    print("done", name, flush=True)


if __name__ == "__main__":
    main()
