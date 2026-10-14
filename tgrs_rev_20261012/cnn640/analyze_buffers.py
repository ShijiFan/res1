"""Statistics for the CNN spatial-isolation sensitivity (../plan/PROTOCOL_CNN640_20261012.md).

Same estimator as merged_tgrs_20260930/plan/substantive_20261002/analyze_cnn.py: per-source confusion averaged
over five seeds, cross-track BA via s11_contrasts.summarise, 20,000 paired block draws (RandomState(42)),
percentile intervals over draws in which all classes are present in both arms.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
SUB = SAR / "merged_tgrs_20260930" / "plan" / "substantive_20261002"
sys.path.insert(0, str(SAR / "A0_gamma0_rerun_20260929" / "scripts"))
from s11_contrasts import summarise  # noqa: E402

SEEDS = [17, 29, 43, 53, 71]
TR = ["t15", "t37", "t88", "t139"]
PATCHES = [1, 3, 5, 9, 11]
used = []


def base(series, p):
    if series == "B400":
        return SUB / "cnn" if p in (1, 3, 5) else HERE / "B400"
    return HERE / "B640"


def family(series, p, arm, full=False):
    x = np.zeros((4, 4, 12, 5, 5))
    for si, src in enumerate(TR):
        for seed in SEEDS:
            f = base(series, p) / f"fixed_p{p}" / f"{arm}_{src}_s{seed}" / (
                "blocks_confusion_full.npy" if full else "blocks_confusion.npy")
            used.append(f)
            x[si] += np.load(f) / len(SEEDS)
    return x


idx = np.random.RandomState(42).randint(0, 12, size=(20000, 12))
w = np.eye(12)[idx].sum(1)
one = np.ones((1, 12))
rows, draws = [], {}
for series in ("B400", "B640"):
    for full in (False, True):
        for p in PATCHES:
            a, b = family(series, p, "CNN_IG", full), family(series, p, "CNN_I", full)
            pa, pb = summarise(a, one)[0][0], summarise(b, one)[0][0]
            aa, _, ok = summarise(a, w)
            bb, _, ok2 = summarise(b, w)
            valid = ok & ok2
            d = np.where(valid, aa - bb, np.nan)
            draws[series, full, p] = d
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append(dict(series=series, eval="full" if full else "sample", patch=p, BA_I=pb, BA_IG=pa,
                             delta=pa - pb, lo=lo, hi=hi, RER_pct=100 * (pa - pb) / (100 - pb),
                             n_valid=int(valid.sum())))
res = pd.DataFrame(rows)
diffs = []
for series in ("B400", "B640"):
    for p, q in ((1, 5), (3, 5), (5, 9), (5, 11)):
        d = draws[series, False, p] - draws[series, False, q]
        pt = res[(res.series == series) & (res["eval"] == "sample") & (res.patch == p)].delta.iloc[0] - \
            res[(res.series == series) & (res["eval"] == "sample") & (res.patch == q)].delta.iloc[0]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        diffs.append(dict(series=series, contrast=f"delta_p{p}-delta_p{q}", point=pt, lo=lo, hi=hi))
    for p in PATCHES:  # buffer effect on the increment
        pass
for p in PATCHES:
    d = draws["B640", False, p] - draws["B400", False, p]
    pt = res[(res.series == "B640") & (res["eval"] == "sample") & (res.patch == p)].delta.iloc[0] - \
        res[(res.series == "B400") & (res["eval"] == "sample") & (res.patch == p)].delta.iloc[0]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    diffs.append(dict(series="B640-B400", contrast=f"delta_p{p}", point=pt, lo=lo, hi=hi))
dif = pd.DataFrame(diffs)
res.to_csv(HERE / "contrasts_buffers.csv", index=False)
dif.to_csv(HERE / "differences_buffers.csv", index=False)
(HERE / "analysis_hashes.json").write_text(json.dumps(
    {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     "inputs": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in dict.fromkeys(used)}}, indent=2))
pd.set_option("display.width", 200)
print(res.round(2).to_string(index=False))
print(dif.round(2).to_string(index=False))
