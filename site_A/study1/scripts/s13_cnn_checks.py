"""CNN supplementary checks C1-C3 (PROTOCOL.md section 5.3).

  --stage cv     C1: 12-config grid, R1's 3-fold spatial CV inside the training blocks (seed 17,
                 13,333 train / 6,667 val pixels per fold, RandomState(17 + fold), 400 m buffer),
                 per arm and source track -> ../ws/cnn_checks/cv_selected.json (+ cv_records.csv)
  --stage final  C1+C2: selected configs, 9x9 patches, seeds 17/29/43/53/71 -> runs_cnn2/tuned_p9/
  --stage patch  C3: fixed section-5.1 hyper-parameters, 5 seeds, patch 5/9/11 (400 m buffer) and
                 15 (600 m buffer, secondary) -> runs_cnn2/fixed_p{5,9,11,15}/
Each run dir holds blocks_confusion.npy (800-pixel sample), blocks_confusion_full.npy (all C2
pixels of the eval blocks), model.pt, norm.npz (train-only channel mean/std), config.json.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_run_R1_v7 as s4  # noqa: E402
from s10_cnn_baseline import ARMS, Net, block_cm, predict  # noqa: E402

r1 = s4.r1
WS = s4.WS
OUT = WS / "cnn_checks"
RUNS = WS / "runs_cnn2"
SEEDS5 = [17, 29, 43, 53, 71]
FIXED = dict(lr=1e-3, epochs=30, wd=1e-4)
GRID = [dict(lr=lr, epochs=ep, wd=wd) for lr, ep, wd in itertools.product([1e-3, 3e-4], [15, 30, 60], [1e-4, 1e-3])]
BS = 256
torch.set_num_threads(12)


def patches(stack, rows, cols, half):
    pad = np.pad(stack, ((0, 0), (half, half), (half, half)), constant_values=np.nan)
    off = np.arange(-half, half + 1)
    rr = rows[:, None, None] + half + off[None, :, None]
    cc = cols[:, None, None] + half + off[None, None, :]
    return np.transpose(pad[:, rr, cc], (1, 0, 2, 3)).astype("float32")


def load():
    sv = r1.SAR_V2
    c7 = np.load(s4.CUBE)
    v4 = np.load(sv / "data" / "flevoland_datacube_v4_multibaseline.npz")
    cm = np.load(sv / "labels" / "consensus_mask.npz")
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om = np.load(s4.SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy")
    vi = np.where(mmu["mmu_valid"])
    omv = om[vi]
    y = v4["y"][vi].astype(np.int64)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c_mask = omv & (y >= 0) & (y < 5) & (y_clc >= 0) & (y_clc < 5) & (y == y_clc)
    gy, gx = v4["grid_y"][vi].astype(int), v4["grid_x"][vi].astype(int)
    bids = (gy // 125) * 100 + (gx // 125)
    coords = np.column_stack([gy, gx]).astype(float) * 40.0
    sm = json.loads((WS / "splits" / "split_manifest.json").read_text())
    valid = c7["four_track_valid"]
    stacks = {arm: {t: np.stack([np.where(valid, c7[f"{t}_{k}"], np.nan) for k in layers]) for t in r1.TR}
              for arm, layers in ARMS.items()}
    return dict(y=y, c_mask=c_mask, gy=gy, gx=gx, bids=bids, coords=coords, sm=sm, stacks=stacks)


def fit(stack, D, idx, cfg, half, seed):
    Xtr = patches(stack, D["gy"][idx], D["gx"][idx], half)
    mu = np.nanmean(Xtr, axis=(0, 2, 3), keepdims=True)
    sd = np.nanstd(Xtr, axis=(0, 2, 3), keepdims=True) + 1e-6
    norm = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")  # noqa: E731
    torch.manual_seed(seed)
    model = Net(stack.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Xt, yt = torch.from_numpy(norm(Xtr)), torch.from_numpy(D["y"][idx])
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            j = perm[i:i + BS]
            opt.zero_grad()
            nn.functional.cross_entropy(model(Xt[j]), yt[j]).backward()
            opt.step()
    return model, norm, mu, sd


def ba(y, p):
    c = confusion_matrix(y, p, labels=[0, 1, 2, 3, 4])
    s = c.sum(1)
    return float(np.mean(np.divide(np.diag(c), s, out=np.zeros(5), where=s > 0)) * 100)


def stage_cv(D):
    OUT.mkdir(parents=True, exist_ok=True)
    f = D["sm"]["source_spatial_cv_3fold"]
    fb = [f["fold_1_blocks"], f["fold_2_blocks"], f["fold_3_blocks"]]
    folds = []
    for k in range(3):
        trb = [b for j in range(3) if j != k for b in fb[j]]
        tr_c = np.where(np.isin(D["bids"], trb) & D["c_mask"])[0]
        va_c = np.where(np.isin(D["bids"], fb[k]) & D["c_mask"])[0]
        d, _ = cKDTree(D["coords"][va_c]).query(D["coords"][tr_c])
        tr_b = tr_c[d >= 400.0]
        rng = np.random.RandomState(17 + k + 1)
        ti = rng.choice(tr_b, min(13333, len(tr_b)), replace=False)
        vi_ = rng.choice(va_c, min(6667, len(va_c)), replace=False)
        assert cKDTree(D["coords"][ti]).query(D["coords"][vi_])[0].min() >= 400.0
        folds.append((ti, vi_))
    recs, sel = [], {}
    for arm in ARMS:
        for src in r1.TR:
            st = D["stacks"][arm][src]
            scores = []
            for ci, cfg in enumerate(GRID):
                bas = []
                for k, (ti, vi_) in enumerate(folds):
                    model, norm, _, _ = fit(st, D, ti, cfg, 4, 17)
                    p = predict(model, norm(patches(st, D["gy"][vi_], D["gx"][vi_], 4)))
                    bas.append(ba(D["y"][vi_], p))
                    recs.append({"arm": arm, "src": src, **cfg, "fold": k + 1, "ba": bas[-1]})
                scores.append((float(np.mean(bas)), ci))
                print(arm, src, cfg, f"mean BA {np.mean(bas):.2f}", flush=True)
            best = max(s for s, _ in scores)
            # tie within 0.1 pp -> fewer epochs, then larger lr (simpler, pre-registered)
            tied = [GRID[ci] for s, ci in scores if best - s <= 0.1]
            choice = sorted(tied, key=lambda c: (c["epochs"], -c["lr"], c["wd"]))[0]
            sel[f"{arm}_{src}"] = {**choice, "cv_mean_ba": round(best, 3)}
            pd.DataFrame(recs).to_csv(OUT / "cv_records.csv", index=False)
    (OUT / "cv_selected.json").write_text(json.dumps(sel, indent=2))


def run_family(D, tag, half, cfg_of, buffer_m):
    sm = D["sm"]["historical_development_split"]
    trb, teb = sm["train_blocks"], sm["eval_blocks"]
    bids, cmask, coords = D["bids"], D["c_mask"], D["coords"]
    ev = np.concatenate([np.random.RandomState(42 + b).choice(np.where((bids == b) & cmask)[0],
                         min(800, int(((bids == b) & cmask).sum())), replace=False) for b in teb])
    ev_full = np.where(np.isin(bids, teb) & cmask)[0]
    tr_c = np.where(np.isin(bids, trb) & cmask)[0]
    d, _ = cKDTree(coords[ev_full]).query(coords[tr_c])
    tr_b = tr_c[d >= buffer_m]
    for arm in ARMS:
        for seed in SEEDS5:
            i_tr = np.random.RandomState(seed).choice(tr_b, min(20000, len(tr_b)), replace=False)
            assert cKDTree(coords[i_tr]).query(coords[ev_full])[0].min() >= buffer_m
            for src in r1.TR:
                od = RUNS / tag / f"{arm}_{src}_s{seed}"
                if (od / "blocks_confusion_full.npy").exists():
                    continue
                t0 = time.time()
                cfg = cfg_of(arm, src)
                model, norm, mu, sd = fit(D["stacks"][arm][src], D, i_tr, cfg, half, seed)
                bc = np.zeros((4, len(teb), 5, 5), dtype=np.int64)
                bcf = np.zeros_like(bc)
                for ti, tgt in enumerate(r1.TARGET_ORDER):
                    st = D["stacks"][arm][tgt]
                    p = predict(model, norm(patches(st, D["gy"][ev], D["gx"][ev], half)))
                    bc[ti] = block_cm(D["y"][ev], p, bids[ev], teb)
                    pf = predict(model, norm(patches(st, D["gy"][ev_full], D["gx"][ev_full], half)))
                    bcf[ti] = block_cm(D["y"][ev_full], pf, bids[ev_full], teb)
                od.mkdir(parents=True, exist_ok=True)
                np.save(od / "blocks_confusion.npy", bc)
                np.save(od / "blocks_confusion_full.npy", bcf)
                np.savez(od / "norm.npz", mu=mu, sd=sd)
                torch.save(model.state_dict(), od / "model.pt")
                (od / "config.json").write_text(json.dumps({"arm": arm, "src": src, "seed": seed, "patch": 2 * half + 1,
                                                            "buffer_m": buffer_m, **cfg, "seconds": round(time.time() - t0, 1)}))
                print(tag, od.name, f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "final", "patch"], required=True)
    a = ap.parse_args()
    D = load()
    if a.stage == "cv":
        stage_cv(D)
    elif a.stage == "final":
        sel = json.loads((OUT / "cv_selected.json").read_text())
        run_family(D, "tuned_p9", 4, lambda arm, src: {k: sel[f"{arm}_{src}"][k] for k in ("lr", "epochs", "wd")}, 400.0)
    else:
        for tag, half, buf in (("fixed_p5", 2, 400.0), ("fixed_p9", 4, 400.0), ("fixed_p11", 5, 400.0), ("fixed_p15", 7, 600.0)):
            run_family(D, tag, half, lambda arm, src: FIXED, buf)


if __name__ == "__main__":
    main()
