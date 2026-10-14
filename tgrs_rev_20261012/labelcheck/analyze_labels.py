"""Reference-label sensitivity, step 2 (../plan/PROTOCOL_LABELS_20261012.md).

Per-block confusion pooled over the 12 cross-track transfers and averaged over seeds; 20,000 paired block draws
(RandomState(42)); percentile intervals over draws in which every reference class is present.
L0 = consensus pixels, WorldCover (reproduces the main design on all C2 pixels); L1a/L1b = all Omega pixels with
WorldCover / CORINE alone; L1c = non-consensus pixels, WorldCover; L2 = BRP 2024 pure grassland / cropland cells
(recall of the two classes and their mean).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows, hashes = [], {}
for tag in ("Site_A", "Site_B"):
    f = HERE / f"{tag}_omega.npz"
    hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    z = np.load(f)
    P, blk, eb = z["preds"], z["block"], z["eval_blocks"]
    nb = len(eb)
    bi = np.searchsorted(np.sort(eb), blk)
    assert np.array_equal(np.sort(eb)[bi], blk)
    w = np.eye(nb)[np.random.RandomState(42).randint(0, nb, size=(20000, nb))].sum(1)
    one = np.ones((1, nb))
    y_wc, y_clc, c2, reg = z["y_wc"], z["y_clc"], z["c2"].astype(bool), z["reg"].astype(int)
    ok5 = lambda y: (y >= 0) & (y < 5)  # noqa: E731
    sets = {"L0": (y_wc, c2 & ok5(y_wc), range(5)), "L1a": (y_wc, ok5(y_wc), range(5)),
            "L1b": (y_clc, ok5(y_clc), range(5)), "L1c": (y_wc, ~c2 & ok5(y_wc), range(5)),
            "L2": (reg, reg > 0, (1, 2))}
    for sname, (y, m, classes) in sets.items():
        classes = list(classes)
        for li, learner in enumerate(("RF", "SVC")):
            res = {}
            for ri in (0, 1):
                # conf[b, true k, pred j] summed over cross transfers, averaged over seeds
                conf = np.zeros((nb, len(classes), 5))
                for si in range(4):
                    for ki in range(3):
                        for ti in range(4):
                            if ti == si:
                                continue
                            p = P[li, ri, si, ki, ti]
                            for kk, k in enumerate(classes):
                                sel = m & (y == k)
                                conf[:, kk] += np.stack([np.bincount(bi[sel & (p == j)], minlength=nb)
                                                         for j in range(5)], 1) / 3
                res[ri] = conf

            def stat(wt, conf):
                c = np.einsum("db,bkj->dkj", wt, conf)
                n = c.sum(2)
                rec = np.stack([c[:, kk, k] for kk, k in enumerate(classes)], 1) / np.maximum(n, 1e-12)
                return rec.mean(1), rec, (n > 0).all(1)
            b0, r0, _ = stat(one, res[0])
            b1, r1, _ = stat(one, res[1])
            d0, _, v0 = stat(w, res[0])
            d1, _, v1 = stat(w, res[1])
            d = np.where(v0 & v1, d1 - d0, np.nan) * 100
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            npx = int(m.sum())
            row = dict(site=tag, set=sname, learner=learner, n_pixels=npx, BA_I=100 * b0[0], BA_IG=100 * b1[0],
                       delta=100 * (b1[0] - b0[0]), lo=lo, hi=hi, n_valid=int((v0 & v1).sum()))
            for kk, k in enumerate(classes):
                row[f"rec{k}_I"], row[f"rec{k}_IG"] = 100 * r0[0, kk], 100 * r1[0, kk]
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(HERE / "label_sensitivity.csv", index=False)
script = Path(__file__)
(HERE / "analysis_hashes.json").write_text(json.dumps(
    {"script": hashlib.sha256(script.read_bytes()).hexdigest(), "inputs": hashes}, indent=2))
pd.set_option("display.width", 250)
print(out.round(2).to_string(index=False))
