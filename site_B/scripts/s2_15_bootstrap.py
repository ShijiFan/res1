"""Site 2, Study 1: paired block-bootstrap contrasts, generalised to any number of evaluation blocks.

Same estimators as A0 run_t1_canonical_results.py / s11_contrasts.py: point estimates are per-seed BA
(macro recall per source-target pair, averaged over the 12 cross-track pairs) averaged over seeds;
intervals use the seed-averaged confusion arrays and 20,000 draws of the evaluation blocks with
RandomState(42), keeping draws in which every class is present in every cross pair.
Usage: python s2_15_bootstrap.py r1 | cnn
Out:   ../runs/STUDY1/ws/tables/{R1_main.csv, R1_contrasts.csv, R1_per_class.csv} or CNN_contrasts.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "runs" / "STUDY1" / "ws"
TR = ["t15", "t37", "t88", "t139"]
N_BOOT = 20000
CLASSES = ["Forest", "Grassland", "Cropland", "Built-up", "Water"]
CROSS, INN = ~np.eye(4, dtype=bool), np.eye(4, dtype=bool)


def load(d, prefix, seeds, fname="blocks_confusion.npy"):
    """-> (n_seeds, 4 src, 4 tgt, NB, 5, 5)"""
    return np.stack([np.stack([np.load(WS / d / f"{prefix}_{s}_s{seed}" / fname) for s in TR]) for seed in seeds]).astype(float)


def ba(cm):
    den = cm.sum(-1)
    rec = np.divide(np.diagonal(cm, axis1=-2, axis2=-1), den, out=np.zeros_like(den), where=den > 0)
    return rec.mean(-1) * 100


def oa(cm):
    return np.diagonal(cm, axis1=-2, axis2=-1).sum(-1) / cm.sum((-2, -1)) * 100


def point(a):
    """per-seed pooled-over-blocks metrics, averaged over seeds -> dict"""
    cm = a.sum(3)  # (S, 4, 4, 5, 5)
    b, o = ba(cm), oa(cm)
    return {"cross_ba": float(b[:, CROSS].mean()), "in_ba": float(b[:, INN].mean()),
            "cross_oa": float(o[:, CROSS].mean()), "in_oa": float(o[:, INN].mean()),
            "cross_ba_seed_sd": float(b[:, CROSS].mean(1).std(ddof=0))}


def boot(a, counts):
    pooled = a.mean(0)  # (4, 4, NB, 5, 5)
    cm = np.einsum("nb,stbkj->nstkj", counts, pooled, optimize=True)
    ok = (cm.sum(-1)[:, CROSS, :] > 0).all(axis=(1, 2))
    b, o = ba(cm), oa(cm)
    return {"cross_ba": b[:, CROSS].mean(1), "in_ba": b[:, INN].mean(1), "cross_oa": o[:, CROSS].mean(1)}, ok


def counts_for(nb):
    idx = np.random.RandomState(42).randint(0, nb, size=(N_BOOT, nb))
    return np.eye(nb)[idx].sum(1)


def contrast(name, A, B, counts, metric="cross_ba", paired_in=False):
    pa, pb = point(A), point(B)
    ba_, oka = boot(A, counts)
    bb_, okb = boot(B, counts)
    ok = oka & okb
    if paired_in:
        pt = (pa["cross_ba"] - pb["cross_ba"]) - (pa["in_ba"] - pb["in_ba"])
        d = ((ba_["cross_ba"] - bb_["cross_ba"]) - (ba_["in_ba"] - bb_["in_ba"]))[ok]
    else:
        pt = pa[metric] - pb[metric]
        d = (ba_[metric] - bb_[metric])[ok]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"contrast": name, "metric": "dcross_minus_din_ba" if paired_in else metric, "A": round(pa[metric], 3),
            "B": round(pb[metric], 3), "delta": round(pt, 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
            "n_valid_boot": int(ok.sum())}


def r1_tables():
    seeds = [17, 29, 43]
    reps = ["I8", "I28", "I8_G9", "F12", "I28_G29"]
    out = WS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fam = {(l, r): load("runs_R1", f"{l}_{r}", seeds) for l in ("RF", "SVC") for r in reps}
    nb = fam[("RF", "I28")].shape[3]
    counts = counts_for(nb)
    pd.DataFrame([{"learner": l, "representation": r, **{k: round(v, 3) for k, v in point(a).items()}}
                  for (l, r), a in fam.items()]).to_csv(out / "R1_main.csv", index=False)
    rows = []
    for l in ("RF", "SVC"):
        g = lambda r: fam[(l, r)]  # noqa: E731
        for name, A, B in (("C-main", g("I28_G29"), g("I28")), ("C-scalar", g("I8_G9"), g("I8")),
                           ("C-baseline", g("I28"), g("I8")), ("C-ablation", g("F12"), g("I8_G9"))):
            rows.append({"learner": l, **contrast(name, A, B, counts)})
        rows.append({"learner": l, **contrast("C-main", g("I28_G29"), g("I28"), counts, "cross_oa")})
        rows.append({"learner": l, **contrast("C-paired", g("I28_G29"), g("I28"), counts, paired_in=True)})
    pd.DataFrame(rows).to_csv(out / "R1_contrasts.csv", index=False)
    pcr = []
    for l in ("RF", "SVC"):
        for r in ("I28", "I28_G29"):
            cm = fam[(l, r)].mean(0).sum(2)[CROSS].sum(0)  # pooled over cross pairs, blocks, seeds
            rec = np.diag(cm) / cm.sum(1) * 100
            conf_forest = cm[:, 0] / cm.sum(1) * 100
            for k, c in enumerate(CLASSES):
                pcr.append({"learner": l, "representation": r, "class": c, "recall": round(float(rec[k]), 2),
                            "pred_as_forest_pct": round(float(conf_forest[k]), 2),
                            "unique_eval_pixels": int(fam[(l, r)][0, 0, 0].sum(0)[k].sum())})
    pd.DataFrame(pcr).to_csv(out / "R1_per_class.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    {"r1": r1_tables}[sys.argv[1]]()
