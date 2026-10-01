"""Paired block-bootstrap contrasts between any two model families (same bootstrap as T1).

A family is a directory pattern "<dir>/<prefix>_{src}_s{seed}/<file>" holding (4 targets, 12 blocks,
5, 5) confusion arrays. BA is pooled as in T1: seeds averaged, per (source, target) 5-class
macro recall, then averaged over the 12 cross-track pairs (and the 4 in-track pairs).
Bootstrap: 20,000 draws of the 12 blocks, RandomState(42), draws missing any class dropped.
Usage: python s11_contrasts.py            (runs the pre-registered CNN / RF contrasts)
Out:   ../ws/tables_cnn/contrasts.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "ws"
TR, SEEDS = ["t15", "t37", "t88", "t139"], [17, 29, 43]
N_BOOT = 20000


def load_family(d, prefix, fname, seeds=SEEDS):
    a = np.zeros((4, 4, 12, 5, 5))
    for si, s in enumerate(TR):
        for seed in seeds:
            a[si] += np.load(WS / d / f"{prefix}_{s}_s{seed}" / fname)
    return a / len(seeds)


def ba(cm):  # cm (..., 5, 5)
    den = cm.sum(-1)
    rec = np.divide(np.diagonal(cm, axis1=-2, axis2=-1), den, out=np.zeros_like(den), where=den > 0)
    return rec.mean(-1) * 100, (den > 0).all(-1)


def summarise(a, counts):
    cm = np.einsum("nb,stbkj->nstkj", counts, a, optimize=True)
    b, ok = ba(cm)
    cross, inn = ~np.eye(4, dtype=bool), np.eye(4, dtype=bool)
    return b[:, cross].mean(1), b[:, inn].mean(1), ok[:, cross].all(1)


def main():
    fam = {
        "RF_I28": ("runs_R1", "RF_I28", "blocks_confusion.npy"),
        "RF_I28_G29": ("runs_R1", "RF_I28_G29", "blocks_confusion.npy"),
        "CNN_I": ("runs_cnn", "CNN_I", "blocks_confusion.npy"),
        "CNN_IG": ("runs_cnn", "CNN_IG", "blocks_confusion.npy"),
        "CNN_I_full": ("runs_cnn", "CNN_I", "blocks_confusion_full.npy"),
        "CNN_IG_full": ("runs_cnn", "CNN_IG", "blocks_confusion_full.npy"),
    }
    pairs = [("CNN_IG", "CNN_I"), ("CNN_I", "RF_I28"), ("CNN_IG", "RF_I28_G29"), ("CNN_IG_full", "CNN_I_full")]
    idx = np.random.RandomState(42).randint(0, 12, size=(N_BOOT, 12))
    counts = np.eye(12)[idx].sum(axis=1)
    point = np.ones((1, 12))
    data = {k: load_family(*v) for k, v in fam.items()}
    rows = []
    for a_name, b_name in pairs:
        pa, pia, _ = summarise(data[a_name], point)
        pb, pib, _ = summarise(data[b_name], point)
        ba_a, _, oka = summarise(data[a_name], counts)
        ba_b, _, okb = summarise(data[b_name], counts)
        ok = oka & okb
        d = (ba_a - ba_b)[ok]
        rows.append({"contrast": f"{a_name} - {b_name}", "A_cross_ba": round(float(pa[0]), 3),
                     "B_cross_ba": round(float(pb[0]), 3), "delta_cross_ba": round(float(pa[0] - pb[0]), 3),
                     "ci_lo": round(float(np.percentile(d, 2.5)), 3), "ci_hi": round(float(np.percentile(d, 97.5)), 3),
                     "A_in_ba": round(float(pia[0]), 3), "B_in_ba": round(float(pib[0]), 3),
                     "n_valid_boot": int(ok.sum())})
    df = pd.DataFrame(rows)
    (WS / "tables_cnn").mkdir(parents=True, exist_ok=True)
    df.to_csv(WS / "tables_cnn" / "contrasts.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
