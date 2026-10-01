"""Site 2, Study 1: autumn replication. Frozen spring R1 models (I28, I28_G29; RF/SVC; 4 sources; 3 seeds)
applied to the October 2024 cube on the spring evaluation pixels that are valid in autumn (as site 1,
tgrs_revision/scripts/run_external_autumn_eval.py). BA intervals are bias-corrected and accelerated
(BCa) block bootstrap, 20,000 draws, acceleration from leave-one-block-out jackknife, as at site 1.
Out: ../runs/STUDY1/ws/runs_autumn/*/blocks_confusion.npy, ../runs/STUDY1/ws/tables/autumn.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_14_study1_R1 as s14  # noqa: E402
import s2_15_bootstrap as bs  # noqa: E402

r1 = s14.r1
WS = s14.WS
OUT = WS / "runs_autumn"


def predict_all():
    ca = dict(np.load(s14.S1 / "cube_autumn.npz"))
    L = np.load(s14.S1 / "labels.npz")
    vi = np.where(L["mmu_valid"])
    sm = json.loads((s14.S1 / "split_manifest.json").read_text())
    teb = sm["historical_development_split"]["eval_blocks"]
    s = pd.read_parquet(WS / "runs_R1" / "RF_I28_t15_s17" / "predictions.parquet")
    s = s[s.target_track == "t15"].sort_values("pixel_index")
    pix, blk, y = s.pixel_index.values, s.block_id.values, s.y_true.values
    ok = ca["four_track_valid"][vi][pix]
    pix, blk, y = pix[ok], blk[ok], y[ok]
    feats, _ = r1.extract_features(ca, vi)
    for k in [k for k in feats if k not in ("I28", "I28_G29")]:
        del feats[k]
    for learner in ("RF", "SVC"):
        for rep in ("I28", "I28_G29"):
            for src in r1.TR:
                for seed in r1.SEEDS:
                    name = f"{learner}_{rep}_{src}_s{seed}"
                    od = OUT / name
                    if (od / "blocks_confusion.npy").exists():
                        continue
                    rd = WS / "runs_R1" / name
                    clf = load(rd / "model.joblib")
                    sc = load(rd / "scaler.joblib") if (rd / "scaler.joblib").exists() else None
                    bc = np.zeros((4, len(teb), 5, 5), dtype=np.int64)
                    for ti, tgt in enumerate(r1.TARGET_ORDER):
                        X = feats[rep][tgt][pix]
                        p = clf.predict(sc.transform(X) if sc is not None else X)
                        for bi, b in enumerate(teb):
                            m = blk == b
                            bc[ti, bi] = confusion_matrix(y[m], p[m], labels=[0, 1, 2, 3, 4])
                    od.mkdir(parents=True, exist_ok=True)
                    np.save(od / "blocks_confusion.npy", bc)
                    print("autumn", name, flush=True)
    return int(ok.sum()), int(len(ok))


def bca(theta_hat, boots, jack, alpha=0.05):
    z0 = norm.ppf(np.clip((boots < theta_hat).mean(), 1e-6, 1 - 1e-6))
    jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6 * (((jm - jack) ** 2).sum()) ** 1.5 + 1e-12)
    q = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        q.append(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) * 100)
    return np.percentile(boots, q)


def cross_ba_present(a, counts):
    """Site-1 autumn rule: BA averaged over the classes present in the draw (not zero-filled)."""
    cm = np.einsum("nb,stbkj->nstkj", counts, a.mean(0), optimize=True)
    den = cm.sum(-1)
    rec = np.where(den > 0, np.diagonal(cm, axis1=-2, axis2=-1) / np.where(den > 0, den, 1), np.nan)
    return np.nanmean(rec, axis=-1)[:, bs.CROSS].mean(1) * 100


def stats(n_used, n_total):
    bs.WS = WS
    rows = []
    for learner in ("RF", "SVC"):
        A = bs.load("runs_autumn", f"{learner}_I28_G29", r1.SEEDS)
        B = bs.load("runs_autumn", f"{learner}_I28", r1.SEEDS)
        nb = A.shape[3]
        pa, pb = bs.point(A), bs.point(B)
        est = pa["cross_ba"] - pb["cross_ba"]
        cnt = bs.counts_for(nb)
        d = cross_ba_present(A, cnt) - cross_ba_present(B, cnt)  # all draws retained (site-1 rule)
        jack = []
        for j in range(nb):
            w = np.ones((1, nb))
            w[0, j] = 0
            jack.append(cross_ba_present(A, w)[0] - cross_ba_present(B, w)[0])
        lo, hi = bca(est, d, np.array(jack))
        oa_d = (bs.boot(A, bs.counts_for(nb))[0]["cross_oa"] - bs.boot(B, bs.counts_for(nb))[0]["cross_oa"])
        rows.append({"learner": learner, "I28_cross_ba": round(pb["cross_ba"], 2), "I28_G29_cross_ba": round(pa["cross_ba"], 2),
                     "delta_ba": round(est, 2), "bca_lo": round(float(lo), 2), "bca_hi": round(float(hi), 2),
                     "delta_oa": round(pa["cross_oa"] - pb["cross_oa"], 2),
                     "oa_lo": round(float(np.percentile(oa_d, 2.5)), 2), "oa_hi": round(float(np.percentile(oa_d, 97.5)), 2),
                     "eval_pixels_valid_in_autumn": n_used, "eval_pixels_spring": n_total})
    df = pd.DataFrame(rows)
    (WS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(WS / "tables" / "autumn.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    stats(*predict_all())
