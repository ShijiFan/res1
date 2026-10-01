"""Step 6: run all A1 classification experiments (E1, E3, E5, E9, E10, E11, E2w). Resumable.

Every configuration writes experiments/{EXP}/{config_id}.npz with out-of-fold predictions aligned to a
parcel order file, and one row in experiments/REGISTRY.csv. Configurations whose npz exists are skipped,
so the script can be killed and restarted at any time.

Usage:
  python a1_s6_experiments.py --exp all            # everything, in priority order
  python a1_s6_experiments.py --exp E1 E11         # selected blocks
  python a1_s6_experiments.py --exp E3 --draws-2025 20 --draws-2024 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a1_common import (BUDGETS, D, G0L, SEED, TARGET_CLASSES, TRACKS, ensure_dirs, get_logger, load_budget_design,
                       now, pair_dates, update_status)

log = get_logger("s6_experiments")
REG_FIELDS = ["config_id", "exp", "year_train", "year_test", "tracks", "M", "draw", "obs", "learner", "variant",
              "control", "window", "n_features", "n_test", "fit_seconds", "prereg_sha256", "path", "utc", "pairs"]


# ------------------------------------------------------------------ data
class Year:
    def __init__(self, year: int):
        self.year = year
        long = pd.read_csv(D["cube"] / f"parcel_long_{year}.csv.gz",
                           usecols=["track", "source_row", "layer", "key", "value"])
        self.tab = {}
        for (trk, lay), g in long.groupby(["track", "layer"]):
            self.tab[(int(trk), lay)] = g.pivot_table(index="source_row", columns="key", values="value", aggfunc="first")
        sp = pd.read_csv(G0L / f"spatial_splits_{year}.csv")
        self.splits = sp[sp["eligible_ge2"] == 1].set_index("source_row")
        self.designs = {k: v for k, v in load_budget_design().items() if k[0] == year}

    def _design(self, track: int, M: int):
        return [f"{pair_dates(p)[2]}_{pair_dates(p)[3]}" for p in self.designs[(self.year, track, M)]]

    def available(self, track: int):
        """Design pairs (M=9) whose coherence and both scenes' VV/VH exist in the cube."""
        coh, vv, vh = (self.tab.get((track, l)) for l in ("COH", "VV", "VH"))
        if coh is None or vv is None or vh is None:
            return []
        return [k for k in self._design(track, 9)
                if k in coh.columns and all(d in vv.columns and d in vh.columns for d in k.split("_"))]

    def pairs(self, track: int, M: int):
        """Nested uniform design; if a design pair is missing (failed job), replace it by the closest available
        pair in time. Returns [] if fewer than M pairs are available (that budget is then skipped)."""
        avail = self.available(track)
        if len(avail) < M:
            return []
        want = self._design(track, M)
        got = [k for k in want if k in avail]
        spare = [k for k in avail if k not in got]
        for k in want:
            if k in avail:
                continue
            t0 = pd.Timestamp(k.split("_")[0])
            spare.sort(key=lambda s: abs((pd.Timestamp(s.split("_")[0]) - t0).days))
            got.append(spare.pop(0))
        return sorted(got)

    def full_keys(self, track: int):
        return self.available(track)

    def parcel_set(self, variant: str, extra_pairs: dict | None = None) -> pd.DataFrame:
        s = self.splits.copy()
        cls_col = "class_permissive" if variant == "permissive" else "class_strict"
        classes = TARGET_CLASSES + (["Other"] if variant == "other8" else [])
        s = s[s[cls_col].isin(classes)]
        ok = pd.Series(True, index=s.index)
        for trk in TRACKS:
            keys = self.full_keys(trk)
            dates = sorted({d for k in keys for d in k.split("_")})
            for lay, cols in (("VV", dates), ("VH", dates), ("COH", keys)):
                t = self.tab.get((trk, lay))
                if t is None:
                    ok &= False
                    continue
                t = t.reindex(index=s.index, columns=cols)
                ok &= np.isfinite(t.values).all(axis=1) & ((t.values > 0).all(axis=1) if lay != "COH" else True)
        if extra_pairs:
            for trk, keys in extra_pairs.items():
                t = self.tab.get((trk, "COH"))
                t = t.reindex(index=s.index, columns=keys) if t is not None else None
                ok &= np.isfinite(t.values).all(axis=1) if t is not None else False
        s = s[ok].copy()
        s["y"] = s[cls_col].map({c: i for i, c in enumerate(classes)}).astype(int)
        return s.sort_index()


def features(Y: Year, idx, tracks, keys_by_track: dict, obs: str, scene_dates_by_track: dict | None = None):
    blocks = []
    for trk in tracks:
        keys = keys_by_track[trk]
        dates = scene_dates_by_track[trk] if scene_dates_by_track else sorted({d for k in keys for d in k.split("_")})
        vv = 10 * np.log10(Y.tab[(trk, "VV")].reindex(index=idx, columns=dates).values)
        vh = 10 * np.log10(Y.tab[(trk, "VH")].reindex(index=idx, columns=dates).values)
        coh = Y.tab[(trk, "COH")].reindex(index=idx, columns=keys).values if keys else np.empty((len(idx), 0))
        if obs == "O1":
            blocks.append(vv)
        elif obs == "O2":
            blocks += [vv, vh, vh - vv]
        elif obs == "O3":
            blocks += [vv, vh, vh - vv, coh]
        elif obs == "O5":
            blocks.append(coh)
        else:
            raise ValueError(obs)
    return np.hstack(blocks)


def learner(name: str):
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1.0, class_weight="balanced", early_stopping=False,
                                              random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
    raise ValueError(name)


# ------------------------------------------------------------------ bookkeeping
class Runner:
    def __init__(self, prereg: str):
        self.prereg = prereg
        self.reg = D["exp"] / "REGISTRY.csv"
        self.done = set()
        if self.reg.exists():
            self.done = {r["config_id"] for r in csv.DictReader(open(self.reg, encoding="utf-8"))}
        self.years = {}

    def year(self, y):
        if y not in self.years:
            self.years[y] = Year(y)
        return self.years[y]

    def parcels_file(self, exp: str, tag: str, ps: pd.DataFrame):
        p = D["exp"] / exp / f"parcels_{tag}.csv"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            ps[["parcel_id", "y", "block_5km", "assigned_test_fold", "pure_pixels_40m"]].to_csv(p)
        return p

    def run(self, cfg: dict, fit_predict):
        cid = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        if cid in self.done:
            return
        t0 = time.time()
        pred, n_feat, n_test, ptag = fit_predict()
        out = D["exp"] / cfg["exp"] / f"{cid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, y_pred=pred.astype(np.int16), parcels_tag=ptag, cfg=json.dumps(cfg))
        row = {k: cfg.get(k, "") for k in REG_FIELDS}
        row.update(config_id=cid, n_features=n_feat, n_test=n_test, fit_seconds=round(time.time() - t0, 1),
                   prereg_sha256=self.prereg, path=str(out.relative_to(D["exp"])), utc=now())
        new = not self.reg.exists()
        with open(self.reg, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(cid)
        log.info("%s %s %.1fs", cfg["exp"], {k: cfg[k] for k in cfg if k not in ("exp", "pairs")}, time.time() - t0)


def cv_predict(Y_tr, ps_tr, X_tr, Y_te, ps_te, X_te, lrn, variant="primary", control="", rng=None):
    """Out-of-fold predictions for the test parcel set (each test parcel predicted by the fold where it is TEST)."""
    pred = np.full(len(ps_te), -1, dtype=int)
    for k in range(6):
        role_tr = ps_tr[f"fold_{k}_role"]
        tr = (role_tr == "TRAIN") | ((role_tr == "EXCLUDED_400M") if variant == "buffer0" else False)
        te = (ps_te[f"fold_{k}_role"] == "TEST").values
        if te.sum() == 0:
            continue
        ytr = ps_tr["y"].values[tr.values]
        if control == "perm":
            ytr = rng.permutation(ytr)
        m = learner(lrn)
        m.fit(X_tr[tr.values], ytr)
        pred[te] = m.predict(X_te[te])
    return pred


# ------------------------------------------------------------------ experiment blocks
def e1(R: Runner, years, learners=("HGB",), budgets=BUDGETS, obs_list=("O1", "O2", "O3", "O5"), variant="primary",
       exp="E1"):
    for y in years:
        Y = R.year(y)
        ps = Y.parcel_set(variant)
        tag = f"{y}_{variant}"
        R.parcels_file(exp, tag, ps)
        for trk in TRACKS:
            for M in budgets:
                keys = Y.pairs(trk, M)
                if not keys:
                    log.warning("E1 %s T%s M=%s: not enough pairs; skipped", y, trk, M)
                    continue
                for obs in obs_list:
                    for lrn in learners:
                        cfg = dict(exp=exp, year_train=y, year_test=y, tracks=str(trk), M=M, draw="uniform", obs=obs,
                                   learner=lrn, variant=variant, pairs=";".join(keys))

                        def fp(Y=Y, ps=ps, keys=keys, obs=obs, lrn=lrn, trk=trk):
                            X = features(Y, ps.index, [trk], {trk: keys}, obs)
                            return cv_predict(Y, ps, X, Y, ps, X, lrn, variant), X.shape[1], len(ps), tag
                        R.run(cfg, fp)


def e3(R: Runner, draws: dict):
    for y, n_draw in draws.items():
        Y = R.year(y)
        ps = Y.parcel_set("primary")
        tag = f"{y}_primary"
        R.parcels_file("E3", tag, ps)
        for trk in TRACKS:
            full = Y.available(trk)
            for M in [m for m in BUDGETS if m < len(full)]:
                combos = list(combinations(range(len(full)), M))
                rng = np.random.default_rng(SEED + 1000 * y + 10 * trk + M)
                pick = [combos[i] for i in rng.choice(len(combos), size=min(n_draw, len(combos)), replace=False)]
                for d, c in enumerate(pick):
                    keys = [full[i] for i in c]
                    for obs in ("O2", "O3"):
                        cfg = dict(exp="E3", year_train=y, year_test=y, tracks=str(trk), M=M, draw=d, obs=obs,
                                   learner="HGB", variant="primary", pairs=";".join(keys))

                        def fp(Y=Y, ps=ps, keys=keys, obs=obs, trk=trk):
                            X = features(Y, ps.index, [trk], {trk: keys}, obs)
                            return cv_predict(Y, ps, X, Y, ps, X, "HGB"), X.shape[1], len(ps), tag
                        R.run(cfg, fp)


def e5(R: Runner, years):
    for y in years:
        Y = R.year(y)
        ps = Y.parcel_set("primary")
        tag = f"{y}_primary"
        R.parcels_file("E5", tag, ps)
        for Mt in (2, 4, 6, 8):
            keys = {trk: Y.pairs(trk, Mt // 2) for trk in TRACKS}
            if not all(keys.values()):
                continue
            for obs in ("O2", "O3"):
                cfg = dict(exp="E5", year_train=y, year_test=y, tracks="37+88", M=Mt, draw="uniform", obs=obs,
                           learner="HGB", variant="primary", pairs=json.dumps(keys))

                def fp(Y=Y, ps=ps, keys=keys, obs=obs):
                    X = features(Y, ps.index, list(TRACKS), keys, obs)
                    return cv_predict(Y, ps, X, Y, ps, X, "HGB"), X.shape[1], len(ps), tag
                R.run(cfg, fp)


def e9(R: Runner):
    for ya, yb in ((2025, 2024), (2024, 2025)):
        A, B = R.year(ya), R.year(yb)
        pa, pb = A.parcel_set("primary"), B.parcel_set("primary")
        tag = f"{yb}_primary"
        R.parcels_file("E9", tag, pb)
        for trk in TRACKS:
            for M in (1, 2, 4, 9):
                ka, kb = A.pairs(trk, M), B.pairs(trk, M)
                if not ka or len(ka) != len(kb):
                    continue
                for obs in ("O2", "O3"):
                    cfg = dict(exp="E9", year_train=ya, year_test=yb, tracks=str(trk), M=M, draw="uniform", obs=obs,
                               learner="HGB", variant="primary", pairs=";".join(ka) + "|" + ";".join(kb))

                    def fp(A=A, B=B, pa=pa, pb=pb, ka=ka, kb=kb, obs=obs, trk=trk):
                        Xa = features(A, pa.index, [trk], {trk: ka}, obs)  # chronological rank alignment
                        Xb = features(B, pb.index, [trk], {trk: kb}, obs)
                        return cv_predict(A, pa, Xa, B, pb, Xb, "HGB"), Xa.shape[1], len(pb), tag
                    R.run(cfg, fp)


def e10(R: Runner):
    e1(R, (2025, 2024), learners=("RF", "LR"), budgets=(1, 2, 4, 9), obs_list=("O2", "O3"), exp="E10")
    for variant in ("permissive", "other8", "buffer0"):
        e1(R, (2025,), budgets=(2, 9), obs_list=("O2", "O3"), variant=variant, exp="E10")


def e11(R: Runner, n_perm=20):
    y = 2025
    Y = R.year(y)
    ps = Y.parcel_set("primary")
    tag = f"{y}_primary"
    R.parcels_file("E11", tag, ps)
    for trk in TRACKS:
        keys = Y.available(trk)
        if not keys:
            continue
        for obs in ("O2", "O3"):
            for r in range(n_perm):
                cfg = dict(exp="E11", year_train=y, year_test=y, tracks=str(trk), M=9, draw=r, obs=obs, learner="HGB",
                           variant="primary", control="perm", pairs=";".join(keys))

                def fp(keys=keys, obs=obs, r=r, trk=trk):
                    X = features(Y, ps.index, [trk], {trk: keys}, obs)
                    rng = np.random.default_rng(SEED + r)
                    return cv_predict(Y, ps, X, Y, ps, X, "HGB", control="perm", rng=rng), X.shape[1], len(ps), tag
                R.run(cfg, fp)
        # test-time date permutation (O2): train on true order, test with scene order permuted
        cfg = dict(exp="E11", year_train=y, year_test=y, tracks=str(trk), M=9, draw="dateperm", obs="O2",
                   learner="HGB", variant="primary", control="dateperm", pairs=";".join(keys))

        def fp_date(keys=keys, trk=trk):
            dates = sorted({d for k in keys for d in k.split("_")})
            perm = list(np.random.default_rng(SEED).permutation(len(dates)))
            Xtr = features(Y, ps.index, [trk], {trk: keys}, "O2", {trk: dates})
            Xte = features(Y, ps.index, [trk], {trk: keys}, "O2", {trk: [dates[i] for i in perm]})
            return cv_predict(Y, ps, Xtr, Y, ps, Xte, "HGB"), Xtr.shape[1], len(ps), tag
        R.run(cfg, fp_date)
    # track-identity probe: can O2 features tell T37 from T88?
    cfg = dict(exp="E11", year_train=y, year_test=y, tracks="37vs88", M=9, draw="trackprobe", obs="O2",
               learner="HGB", variant="primary", control="trackprobe")

    def fp_probe():
        n = min(len(Y.available(t)) for t in TRACKS)
        X = np.vstack([features(Y, ps.index, [t], {t: Y.available(t)[:n]}, "O2") for t in TRACKS])
        pp = pd.concat([ps, ps])
        pp = pp.assign(y=np.r_[np.zeros(len(ps), int), np.ones(len(ps), int)])
        return cv_predict(Y, pp, X, Y, pp, X, "HGB"), X.shape[1], len(pp), f"{y}_trackprobe"
    pp_file = D["exp"] / "E11" / f"parcels_{y}_trackprobe.csv"
    if not pp_file.exists():
        pp = pd.concat([ps, ps]).assign(y=np.r_[np.zeros(len(ps), int), np.ones(len(ps), int)])
        pp[["parcel_id", "y", "block_5km", "assigned_test_fold", "pure_pixels_40m"]].to_csv(pp_file)
    R.run(cfg, fp_probe)


def e2w(R: Runner):
    """Window experiment on closure subset B2 (and the P0 6-day pilot pair)."""
    b2 = pd.read_csv(D["submit"] / "CLOSURE_SUBSET_B2.csv")
    wins = []
    for _, r in b2.iterrows():
        d1, d2 = r["planned_pair"].split("_")
        legs = r["new_pairs"].split(";")                 # [(d2, d3), (d1, d3)]
        wins.append((int(r["year"]), int(r["track"]), d1, d2, legs[0], legs[1]))
    wins.append((2025, 88, "2025-06-15", "2025-06-27", "2025-06-27_2025-07-03", None))   # P0 pilot 6-day
    for (y, trk, d1, d2, leg_next, leg_long) in wins:
        Y = R.year(y)
        base = f"{d1}_{d2}"
        extra = [k for k in (base, leg_next, leg_long) if k]
        if not all(k in Y.tab[(trk, "COH")].columns for k in extra):
            log.warning("window %s %s %s: missing coherence layers; skipped", y, trk, base)
            continue
        ps = Y.parcel_set("primary", extra_pairs={trk: extra})
        wtag = f"{y}_T{trk}_{d1}"
        R.parcels_file("E2w", wtag, ps)
        variants = {"W0": [], "W1_c12": [base], "W3_next": [leg_next], "W5_c12_next": [base, leg_next]}
        if leg_long:
            variants.update({"W2_long": [leg_long], "W4_all": [base, leg_next, leg_long]})
        for vname, coh_keys in variants.items():
            cfg = dict(exp="E2w", year_train=y, year_test=y, tracks=str(trk), M=1, draw="window", obs=vname,
                       learner="HGB", variant="primary", window=wtag, pairs=";".join(coh_keys))

            def fp(Y=Y, ps=ps, coh_keys=coh_keys, trk=trk, d1=d1, d2=d2, wtag=wtag):
                Xi = features(Y, ps.index, [trk], {trk: []}, "O2", {trk: [d1, d2]})
                Xc = Y.tab[(trk, "COH")].reindex(index=ps.index, columns=coh_keys).values if coh_keys else np.empty((len(ps), 0))
                X = np.hstack([Xi, Xc])
                return cv_predict(Y, ps, X, Y, ps, X, "HGB"), X.shape[1], len(ps), wtag
            R.run(cfg, fp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"])
    ap.add_argument("--draws-2025", type=int, default=20)
    ap.add_argument("--draws-2024", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args()
    ensure_dirs()
    shaf = D["reports"] / "A1_PREREG.sha256"
    if not shaf.exists():
        raise SystemExit("STOP: run a1_s5_prereg.py first")
    R = Runner(shaf.read_text().strip())
    order = ["E1", "E11", "E9", "E5", "E2w", "E10", "E3"] if a.exp == ["all"] else a.exp
    for e in order:
        update_status("s6_experiments", "RUNNING", block=e)
        if e == "E1":
            e1(R, (2025, 2024))
        elif e == "E3":
            e3(R, {2025: a.draws_2025, 2024: a.draws_2024})
        elif e == "E5":
            e5(R, (2025, 2024))
        elif e == "E9":
            e9(R)
        elif e == "E10":
            e10(R)
        elif e == "E11":
            e11(R, n_perm=a.n_perm)
        elif e == "E2w":
            e2w(R)
    update_status("s6_experiments", "DONE", blocks=order)


if __name__ == "__main__":
    main()
