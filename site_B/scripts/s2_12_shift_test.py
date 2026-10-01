"""Site 2, Study 1: label-to-SAR alignment test for WorldCover (PREREG_SITE2.md deviation note D1).

Rule fixed before running: RF (100 trees) on T15-spring I8+g12 features, 20,000 Omega pixels with a valid
WorldCover label, 2-fold spatial split by 5 km block parity, 5 seeds; WorldCover shifted by dx, dy in
{-1, 0, +1} cells. A non-zero shift is adopted ONLY if it beats zero shift by >= 0.5 pp BA in all 5 seeds.
Out: ../runs/STUDY1/shift_test.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "runs" / "STUDY1"
sys.path.insert(0, str(ROOT.parent / "res1_clean" / "src"))
import run_R1_buffered_cv_retrain as r1  # noqa: E402


def main():
    c = dict(np.load(S1 / "cube_spring.npz"))
    L = np.load(S1 / "labels.npz")
    wc = L["y"].astype(int)
    omega = L["omega"]
    vi = np.where(omega)
    feats, _ = r1.extract_features(c, vi)
    X = feats["I8_G9"]["t15"]
    gy, gx = L["grid_y"][vi], L["grid_x"][vi]
    par = ((gy // 125) + (gx // 125)) % 2
    res = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = nd_shift(wc, [dy, dx], cval=-1, order=0, mode="constant")[vi]
            scores = []
            for seed in range(5):
                rng = np.random.RandomState(seed)
                ok = np.where(ys >= 0)[0]
                idx = rng.choice(ok, min(20000, ok.size), replace=False)
                ba = []
                for f in (0, 1):
                    tr, te = idx[par[idx] != f], idx[par[idx] == f]
                    m = RandomForestClassifier(100, n_jobs=6, random_state=seed).fit(X[tr], ys[tr])
                    ba.append(balanced_accuracy_score(ys[te], m.predict(X[te])))
                scores.append(float(np.mean(ba)) * 100)
            res[f"dy{dy:+d}_dx{dx:+d}"] = scores
            print(dy, dx, np.round(scores, 2), flush=True)
    base = np.array(res["dy+0_dx+0"])
    better = {k: bool(np.all(np.array(v) - base >= 0.5)) for k, v in res.items() if k != "dy+0_dx+0"}
    adopt = [k for k, b in better.items() if b]
    choice = max(adopt, key=lambda k: np.mean(res[k])) if adopt else "dy+0_dx+0"
    out = {"rule": "adopt non-zero shift only if >= 0.5 pp BA better than zero shift in all 5 seeds",
           "ba_by_shift": res, "beats_zero_in_all_seeds": better, "adopted_shift": choice}
    (S1 / "shift_test.json").write_text(json.dumps(out, indent=2))
    print("adopted:", choice)


if __name__ == "__main__":
    main()
