"""EXPLORATORY (post hoc, PROTOCOL.md 5.4): BAG built-up recall by cover threshold and building-cluster size.

Pixel sets inside the 12 evaluation blocks (all Omega pixels):
  thr25 / thr50 / thr75 : f_BAG >= 0.25 / 0.50 / 0.75
  iso / small / settle  : f_BAG >= 0.25 AND the 8-connected cluster of f_BAG > 0 pixels has
                          1-4 / 5-50 / > 50 pixels
  far                   : no building within the 5x5 neighbourhood (false built-up rate)
Families: RF and SVC (I28 vs I28_G29, 3 seeds, ws/runs_R1) and CNN (CNN_I vs CNN_IG,
tuned_p9, 5 seeds, ws/runs_cnn2). Differences: 20,000 block-bootstrap draws, RandomState(42).
Out: ../ws/bag_check/sensitivity_results.csv, sensitivity_counts.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from scipy.ndimage import label, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_run_R1_v7 as s4  # noqa: E402
import s13_cnn_checks as s13  # noqa: E402
from s10_cnn_baseline import ARMS, Net, predict  # noqa: E402

r1 = s4.r1
WS = s4.WS
OUT = WS / "bag_check"
URBAN = 3
SETS = ["thr25", "thr50", "thr75", "iso", "small", "settle", "far"]


def main():
    f_bag = np.load(s4.ROOT / "labels" / "f_bag_40m.npy")
    lab, n = label(f_bag > 0, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel())
    csize = size[lab]
    csize[lab == 0] = 0
    D = s13.load()
    sv = r1.SAR_V2
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om = np.load(s4.SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy")
    vi = np.where(mmu["mmu_valid"])
    omv = om[vi]
    teb = D["sm"]["historical_development_split"]["eval_blocks"]
    ev = np.where(np.isin(D["bids"], teb) & omv)[0]
    gy, gx = D["gy"][ev], D["gx"][ev]
    fb, cs = f_bag[gy, gx], csize[gy, gx]
    far = (uniform_filter((f_bag > 0).astype("float32"), 5, mode="constant") == 0)[gy, gx]
    masks = np.stack([fb >= 0.25, fb >= 0.5, fb >= 0.75,
                      (fb >= 0.25) & (cs <= 4), (fb >= 0.25) & (cs > 4) & (cs <= 50), (fb >= 0.25) & (cs > 50), far])
    blk = np.stack([D["bids"][ev] == b for b in teb])  # (NB, N)
    n_set = (blk[:, None, :] & masks[None]).sum(-1)  # (NB, K)
    info = {"n_pixels": int(ev.size), "n_clusters": int(n), "set_sizes": dict(zip(SETS, n_set.sum(0).tolist()))}
    print(info)

    def hits(pu):
        return (blk[:, None, :] & masks[None] & pu[None, None, :]).sum(-1)  # (NB, K)

    counts = {}
    c7 = np.load(s4.CUBE)
    feats, _ = r1.extract_features(c7, vi)
    for k in [k for k in feats if k not in ("I28", "I28_G29")]:
        del feats[k]
    for learner in ("RF", "SVC"):
        for rep in ("I28", "I28_G29"):
            arr = np.zeros((4, 4, 3, len(teb), len(SETS)), dtype=np.int64)
            for si, src in enumerate(r1.TR):
                for ki, seed in enumerate(r1.SEEDS):
                    rd = WS / "runs_R1" / f"{learner}_{rep}_{src}_s{seed}"
                    clf = load(rd / "model.joblib")
                    sc = load(rd / "scaler.joblib") if (rd / "scaler.joblib").exists() else None
                    for ti, tgt in enumerate(r1.TARGET_ORDER):
                        X = feats[rep][tgt][ev]
                        arr[si, ti, ki] = hits(clf.predict(sc.transform(X) if sc is not None else X) == URBAN)
                    print("done", learner, rep, src, seed, flush=True)
            counts[f"{learner}_{rep}"] = arr
    del feats
    for arm in ARMS:
        arr = np.zeros((4, 4, len(s13.SEEDS5), len(teb), len(SETS)), dtype=np.int64)
        for si, src in enumerate(r1.TR):
            for ki, seed in enumerate(s13.SEEDS5):
                rd = WS / "runs_cnn2" / "tuned_p9" / f"{arm}_{src}_s{seed}"
                model = Net(len(ARMS[arm]))
                model.load_state_dict(torch.load(rd / "model.pt"))
                nz = np.load(rd / "norm.npz")
                for ti, tgt in enumerate(r1.TARGET_ORDER):
                    X = s13.patches(D["stacks"][arm][tgt], gy, gx, 4)
                    X = np.nan_to_num((X - nz["mu"]) / nz["sd"], nan=0.0).astype("float32")
                    arr[si, ti, ki] = hits(predict(model, X) == URBAN)
                print("done", arm, src, seed, flush=True)
        counts[arm] = arr
    np.savez(OUT / "sensitivity_counts.npz", n_set=n_set, **counts)

    boot = np.eye(len(teb))[np.random.RandomState(42).randint(0, len(teb), size=(20000, len(teb)))].sum(1)
    cross = ~np.eye(4, dtype=bool)
    rows = []
    for base, aug, fam in (("RF_I28", "RF_I28_G29", "RF"), ("SVC_I28", "SVC_I28_G29", "SVC"), ("CNN_I", "CNN_IG", "CNN")):
        def rate(key, w):
            h = counts[key][cross].mean(axis=1).sum(axis=0)  # (NB, K): seeds averaged, cross pairs summed
            return (w @ h) / np.maximum(w @ (n_set * cross.sum()), 1)
        for k, s in enumerate(SETS):
            r0, r1_ = rate(base, np.ones((1, len(teb))))[0, k], rate(aug, np.ones((1, len(teb))))[0, k]
            d = (rate(aug, boot)[:, k] - rate(base, boot)[:, k]) * 100
            rows.append({"family": fam, "set": s, "n_pixels": int(n_set[:, k].sum()), "base_pct": round(float(r0) * 100, 2),
                         "coh_pct": round(float(r1_) * 100, 2), "delta_pp": round(float(r1_ - r0) * 100, 2),
                         "ci_lo": round(float(np.percentile(d, 2.5)), 2), "ci_hi": round(float(np.percentile(d, 97.5)), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity_results.csv", index=False)
    (OUT / "sensitivity_info.json").write_text(json.dumps(info, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
