"""CNN spatial-isolation sensitivity (../plan/PROTOCOL_CNN640_20261012.md).

Series B400: 400 m buffer, saved training indices of substantive_20261002, CUDA runs for patches 9 and 11
            (patches 1/3/5 already exist with CUDA and are reused in the analysis).
Series B640: 640 m buffer, new training indices (same sampling rule), CUDA runs for patches 1/3/5/9/11.
Usage: python run_cnn_buffers.py B400 | B640
"""
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
SUB = SAR / "merged_tgrs_20260930" / "plan" / "substantive_20261002"
sys.path.insert(0, str(SUB))
import run_small_cnn as rs  # noqa: E402  (sets deterministic CUDA flags, imports s13_cnn_checks as rs.old)

old = rs.old
SERIES = {"B400": dict(buffer=400.0, patches=[9, 11], reuse_idx=True),
          "B640": dict(buffer=640.0, patches=[1, 3, 5, 9, 11], reuse_idx=False)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(series):
    cfg = SERIES[series]
    out = HERE / series
    out.mkdir(exist_ok=True)
    D = old.load()
    sm = D["sm"]["historical_development_split"]
    teb, trb = sm["eval_blocks"], sm["train_blocks"]
    bids, mask, coords = D["bids"], D["c_mask"], D["coords"]
    ev = np.concatenate([np.random.RandomState(42 + b).choice(np.where((bids == b) & mask)[0],
                         min(800, int(((bids == b) & mask).sum())), replace=False) for b in teb])
    ref = np.load(SUB / "cnn" / "evaluation_order.npz")
    assert np.array_equal(ev, ref["sample"]), "evaluation sample differs from substantive_20261002"
    full = np.where(np.isin(bids, teb) & mask)[0]
    cand = np.where(np.isin(bids, trb) & mask)[0]
    dist = cKDTree(coords[full]).query(coords[cand])[0]
    train = cand[dist >= cfg["buffer"]]
    audit, idxs = [], {}
    for seed in old.SEEDS5:
        if cfg["reuse_idx"]:
            idx = np.load(SUB / "cnn" / f"train_indices_s{seed}.npy")
            assert np.isin(idx, train).all()
        else:
            idx = np.random.RandomState(seed).choice(train, min(20000, len(train)), replace=False)
        np.save(out / f"train_indices_s{seed}.npy", idx)
        idxs[seed] = idx
        assert cKDTree(coords[idx]).query(coords[full])[0].min() >= cfg["buffer"]
        tree = cKDTree(np.column_stack([D["gy"][idx], D["gx"][idx]]))
        for p in [1, 3, 5, 9, 11]:
            near = tree.query_ball_point(np.column_stack([D["gy"][full], D["gx"][full]]), p - 1, p=np.inf,
                                         return_length=True)
            audit.append({"seed": seed, "patch": p, "eval_centers_with_shared_input": int((near > 0).sum()),
                          "eval_centers": int(len(full)), "train_candidates": int(len(train))})
    (out / "overlap_audit.json").write_text(json.dumps(audit, indent=2))
    if series == "B640":
        shared = [a for a in audit if a["eval_centers_with_shared_input"] > 0]
        if shared:
            raise SystemExit(f"STOP: shared inputs at 640 m: {shared[:3]}")
    man = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "torch": torch.__version__,
           "device": torch.cuda.get_device_name(), "series": series, **cfg,
           "sha256": {str(p): sha(p) for p in [Path(__file__), Path(rs.__file__), Path(old.__file__),
                                                old.s4.CUBE, HERE.parent / "plan" / "PROTOCOL_CNN640_20261012.md"]}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    t0 = time.time()
    for p in cfg["patches"]:
        for arm in old.ARMS:
            for seed in old.SEEDS5:
                for src in old.r1.TR:
                    od = out / f"fixed_p{p}" / f"{arm}_{src}_s{seed}"
                    od.mkdir(parents=True, exist_ok=True)
                    if (od / "config.json").exists():
                        continue
                    t = time.time()
                    net, norm, mu, sd = rs.fit(D["stacks"][arm][src], D, idxs[seed], old.FIXED, p // 2, seed)
                    bc, bcf = [], []
                    for tgt in old.r1.TARGET_ORDER:
                        st = D["stacks"][arm][tgt]
                        pp = rs.predict(net, norm(old.patches(st, D["gy"][ev], D["gx"][ev], p // 2)))
                        pf = rs.predict(net, norm(old.patches(st, D["gy"][full], D["gx"][full], p // 2)))
                        bc.append(old.block_cm(D["y"][ev], pp, bids[ev], teb))
                        bcf.append(old.block_cm(D["y"][full], pf, bids[full], teb))
                    np.save(od / "blocks_confusion.npy", bc)
                    np.save(od / "blocks_confusion_full.npy", bcf)
                    (od / "config.json").write_text(json.dumps({"patch": p, "arm": arm, "source": src, "seed": seed,
                                                                 **old.FIXED, "batch": old.BS, "buffer_m": cfg["buffer"],
                                                                 "n_train": int(len(idxs[seed])),
                                                                 "seconds": time.time() - t}, indent=2))
                    print(series, p, arm, seed, src, round(time.time() - t, 1), flush=True)
    print("COMPLETE", series, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
