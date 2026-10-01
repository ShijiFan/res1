"""Temporal deep-learning baseline (TempCNN) for Study 2, per PROTOCOL.md (frozen, sha256 in PROTOCOL.sha256).

Reuses each site's frozen Study-2 code (a1_s6_experiments.Year: parcels, classes, 6 spatial folds with the
400 m exclusion, parcel-level linear-power means) and the exact E3 date designs used by HGB (first three
draws per year/track/M in the site's E3 registry), so DL and HGB are compared on identical inputs.
Usage: <yolo11marin python> tdl_run.py --site 1|2
Out:   site{n}/preds/<config>.npz (y_pred aligned with the site's parcel set), site{n}/REGISTRY.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SAR = HERE.parent
SITE_SCRIPTS = {1: SAR / "A1_observation_budget_20260923" / "scripts", 2: SAR / "site2_20260930" / "scripts"}
BUDGETS, N_DRAWS, SEED = (1, 2, 4, 8), 3, 20261001
EPOCHS, PATIENCE, LR, WD, BS = 60, 10, 1e-3, 1e-4, 128
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TempCNN(nn.Module):
    def __init__(self, c_in, length, n_cls=7):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.2))
        self.conv = nn.Sequential(blk(c_in, 64), blk(64, 64), blk(64, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * length, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(self.conv(x))


def sequences(Y, idx, trk, keys, obs):
    """(n, C, 2M): per scene VV dB, VH dB, VH-VV, DOY/365 (+ coherence of the scene's pair for O3)."""
    dates = sorted({d for k in keys for d in k.split("_")})
    vv = 10 * np.log10(Y.tab[(trk, "VV")].reindex(index=idx, columns=dates).values)
    vh = 10 * np.log10(Y.tab[(trk, "VH")].reindex(index=idx, columns=dates).values)
    doy = np.array([pd.Timestamp(d).dayofyear / 365.0 for d in dates])
    ch = [vv, vh, vh - vv, np.broadcast_to(doy, vv.shape)]
    if obs == "O3":
        coh_pair = Y.tab[(trk, "COH")].reindex(index=idx, columns=keys).values
        pos = {d: i for i, k in enumerate(keys) for d in k.split("_")}
        ch.append(coh_pair[:, [pos[d] for d in dates]])
    return np.stack(ch, axis=1).astype("float32")


def ba(y, p, n=7):
    cm = np.zeros((n, n))
    np.add.at(cm, (y, p), 1)
    s = cm.sum(1)
    return float(np.mean(np.divide(np.diag(cm), s, out=np.zeros(n), where=s > 0)))


def fit_predict(X, y, ps, rng_seed):
    pred = np.full(len(y), -1, dtype=int)
    for k in range(6):
        role = ps[f"fold_{k}_role"].values
        tr, te = role == "TRAIN", role == "TEST"
        if te.sum() == 0:
            continue
        blocks = np.unique(ps["block_5km"].values[tr])
        val_blocks = set(np.random.RandomState(SEED).permutation(blocks)[:max(1, int(round(0.1 * len(blocks))))])
        va = tr & ps["block_5km"].isin(val_blocks).values
        fit = tr & ~va
        mu = np.nanmean(X[fit], axis=(0, 2), keepdims=True)
        sd = np.nanstd(X[fit], axis=(0, 2), keepdims=True) + 1e-6
        Z = np.nan_to_num((X - mu) / sd).astype("float32")
        torch.manual_seed(rng_seed)
        model = TempCNN(X.shape[1], X.shape[2]).to(DEV)
        cnt = np.bincount(y[fit], minlength=7).astype(float)
        w = torch.tensor(np.where(cnt > 0, cnt.sum() / (7 * np.maximum(cnt, 1)), 0.0), dtype=torch.float32, device=DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        Xf, yf = torch.tensor(Z[fit], device=DEV), torch.tensor(y[fit], device=DEV)
        Xv, Xt = torch.tensor(Z[va], device=DEV), torch.tensor(Z[te], device=DEV)
        g = torch.Generator(device="cpu").manual_seed(rng_seed)
        best, best_state, bad = -1.0, None, 0
        for _ in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(Xf), generator=g).to(DEV)
            for i in range(0, len(Xf), BS):
                j = perm[i:i + BS]
                if len(j) < 2:
                    continue
                opt.zero_grad()
                nn.functional.cross_entropy(model(Xf[j]), yf[j], weight=w).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = ba(y[va], model(Xv).argmax(1).cpu().numpy())
            if v > best:
                best, bad = v, 0
                best_state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred[te] = model(Xt).argmax(1).cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(SITE_SCRIPTS[a.site]))
    import a1_s6_experiments as s6  # noqa: E402  (site-specific paths via that folder's a1_common)
    out = HERE / f"site{a.site}"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    reg_p = out / "REGISTRY.csv"
    done = set(pd.read_csv(reg_p).config_id) if reg_p.exists() else set()
    hreg = pd.read_csv(s6.D["exp"] / "REGISTRY.csv", dtype={"draw": str})
    hreg = hreg[(hreg.exp == "E3") & (hreg.learner == "HGB") & (hreg.variant == "primary") & hreg.M.isin(BUDGETS)]
    print("device", DEV, "| site", a.site, "| E3 rows", len(hreg), flush=True)
    for year in sorted(hreg.year_test.unique(), reverse=True):
        Y = s6.Year(int(year))
        ps = Y.parcel_set("primary")
        y = ps["y"].values.astype(int)
        for trk in s6.TRACKS:
            for M in BUDGETS:
                sub = hreg[(hreg.year_test == year) & (hreg.tracks.astype(str) == str(trk)) & (hreg.M == M) & (hreg.obs == "O3")]
                draws = sorted(sub.draw.unique(), key=lambda d: int(d))[:N_DRAWS]
                for d in draws:
                    keys = sub[sub.draw == d].pairs.iloc[0].split(";")
                    for obs in ("O2", "O3"):
                        cid = f"{year}_T{trk}_M{M}_d{d}_{obs}"
                        if cid in done:
                            continue
                        hrow = hreg[(hreg.year_test == year) & (hreg.tracks.astype(str) == str(trk)) & (hreg.M == M)
                                    & (hreg.obs == obs) & (hreg.draw == d)]
                        assert len(hrow) >= 1 and hrow.pairs.iloc[0] == ";".join(keys), cid
                        t0 = time.time()
                        X = sequences(Y, ps.index, trk, keys, obs)
                        pred = fit_predict(X, y, ps, SEED)
                        np.savez_compressed(out / "preds" / f"{cid}.npz", y_pred=pred.astype(np.int16))
                        row = {"config_id": cid, "year": year, "track": trk, "M": M, "draw": d, "obs": obs,
                               "pairs": ";".join(keys), "hgb_npz": hrow.path.iloc[0], "n_test": len(y),
                               "ba_dl": round(ba(y, pred), 5), "seconds": round(time.time() - t0, 1)}
                        new = not reg_p.exists()
                        with open(reg_p, "a", newline="", encoding="utf-8") as f:
                            wr = csv.DictWriter(f, fieldnames=list(row))
                            if new:
                                wr.writeheader()
                            wr.writerow(row)
                        done.add(cid)
                        print(cid, row["ba_dl"], f"{row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
