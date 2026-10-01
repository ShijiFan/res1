"""Step 7: statistics for all A1 experiments (paired 5 km block bootstrap), tables and hypothesis checks.

Reads experiments/REGISTRY.csv and the npz predictions; writes CSV tables to RUN/stats and
RUN/stats/HYPOTHESES.json. Pure post-processing: safe to re-run at any time (e.g. while E3 is still running).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from a1_common import BUDGETS, D, TARGET_CLASSES, TRACKS, ensure_dirs, get_logger, now, update_status, write_json

log = get_logger("s7_stats")


# ------------------------------------------------------------------ core metrics
def block_cm(y, p, blocks, C):
    ub, bi = np.unique(blocks, return_inverse=True)
    cm = np.zeros((len(ub), C, C))
    ok = p >= 0
    np.add.at(cm, (bi[ok], y[ok], p[ok]), 1)
    return ub, cm


def ba_from(cmw):
    """cmw: (..., C, C) -> balanced accuracy over classes with support."""
    tp = np.diagonal(cmw, axis1=-2, axis2=-1)
    sup = cmw.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
    return np.nanmean(rec, axis=-1)


def f1_from(cmw):
    tp = np.diagonal(cmw, axis1=-2, axis2=-1)
    sup = cmw.sum(-1)
    pr = cmw.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(pr > 0, tp / pr, np.nan)
        f1 = np.where((rec + prec) > 0, 2 * rec * prec / (rec + prec), 0.0)
    f1 = np.where(sup > 0, f1, np.nan)
    return rec, f1


class Store:
    def __init__(self, B: int, B_head: int):
        self.reg = pd.read_csv(D["exp"] / "REGISTRY.csv", dtype=str)
        self.B, self.B_head = B, B_head
        self.parcels, self.W = {}, {}
        self.cache = {}

    def parcels_of(self, exp, tag):
        k = (exp, tag)
        if k not in self.parcels:
            self.parcels[k] = pd.read_csv(D["exp"] / exp / f"parcels_{tag}.csv", index_col=0)
        return self.parcels[k]

    def weights(self, blocks_sorted, B):
        key = (tuple(blocks_sorted), B)
        if key not in self.W:
            seed = int(hashlib.sha1(("|".join(map(str, blocks_sorted))).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            n = len(blocks_sorted)
            self.W[key] = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
        return self.W[key]

    def get(self, cid, B=None, subset=None):
        """Returns dict: point BA, boot BA (B,), cm (blocks,C,C), block ids."""
        B = B or self.B
        key = (cid, B, subset)
        if key in self.cache:
            return self.cache[key]
        r = self.reg[self.reg.config_id == cid].iloc[0]
        z = np.load(D["exp"] / r["path"], allow_pickle=True)
        tag = str(z["parcels_tag"])
        ps = self.parcels_of(r["exp"], tag)
        y, p = ps["y"].values.astype(int), z["y_pred"].astype(int)
        blocks = ps["block_5km"].astype(str).values
        if subset == "pure4":
            keep = ps["pure_pixels_40m"].values >= 4
            y, p, blocks = y[keep], p[keep], blocks[keep]
        C = int(max(y.max(), p.max()) + 1)
        C = max(C, len(TARGET_CLASSES)) if r["control"] != "trackprobe" else 2
        ub, cm = block_cm(y, p, blocks, C)
        W = self.weights(sorted(ub), B)
        cmw = np.einsum("bn,nij->bij", W, cm)
        out = {"cm": cm, "blocks": ub, "point": float(ba_from(cm.sum(0))), "boot": ba_from(cmw),
               "cmw": cmw, "n": int((p >= 0).sum()), "oa": float(np.trace(cm.sum(0)) / cm.sum())}
        self.cache[key] = out
        return out

    def find(self, **kw):
        q = self.reg
        for k, v in kw.items():
            q = q[q[k].astype(str) == str(v)]
        return q


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a.size else (np.nan, np.nan)


def pval(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if d.size else np.nan


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def sat(M, a, b, k):
    return a - (a - b) * np.exp(-(M - 1) / k)


# ------------------------------------------------------------------ tables
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--B-head", type=int, default=10000)
    a = ap.parse_args()
    ensure_dirs()
    S = Store(a.B, a.B_head)
    out = D["stats"]
    res = {}

    # ---- E1 curves and coherence gain
    curve, dcs = [], []
    for y in (2025, 2024):
        for trk in TRACKS:
            ps_rows = []
            for M in BUDGETS:
                recs = {}
                for obs in ("O1", "O2", "O3", "O5"):
                    q = S.find(exp="E1", year_train=y, tracks=trk, M=M, obs=obs, learner="HGB", variant="primary")
                    if len(q):
                        recs[obs] = S.get(q.iloc[0].config_id, B=a.B_head)
                        g = recs[obs]
                        _, f1 = f1_from(g["cm"].sum(0))
                        lo, hi = ci(g["boot"])
                        curve.append(dict(year=y, track=trk, M=M, N=2 * M, obs=obs, BA=g["point"], lo=lo, hi=hi,
                                          macroF1=float(np.nanmean(f1)), OA=g["oa"], n_test=g["n"]))
                if "O2" in recs and "O3" in recs:
                    d = recs["O3"]["boot"] - recs["O2"]["boot"]
                    lo, hi = ci(d)
                    row = dict(year=y, track=trk, M=M, N=2 * M, dC=recs["O3"]["point"] - recs["O2"]["point"],
                               lo=lo, hi=hi, p=pval(d))
                    if "O1" in recs:
                        dp = recs["O2"]["boot"] - recs["O1"]["boot"]
                        row.update(dPol=recs["O2"]["point"] - recs["O1"]["point"], dPol_lo=ci(dp)[0], dPol_hi=ci(dp)[1])
                    ps_rows.append(row)
            if ps_rows:
                adj = holm([r["p"] for r in ps_rows])
                for r, pa in zip(ps_rows, adj):
                    r["p_holm"] = pa
                dcs += ps_rows
    curve, dcs = pd.DataFrame(curve), pd.DataFrame(dcs)
    curve.to_csv(out / "T_E1_curve.csv", index=False)
    dcs.to_csv(out / "T_E1_deltaC.csv", index=False)

    # ---- N*, equivalent multiplier, equal cost, saturating fits
    nstar, equiv, eqcost, fits = [], [], [], []
    for (y, trk), g in dcs.groupby(["year", "track"]) if len(dcs) else []:
        g = g.sort_values("M")
        ok = (g["hi"] < 0.01).values
        ns = None
        for i in range(len(g)):
            if ok[i:].all():
                ns = int(g["M"].iloc[i])
                break
        first = g[g["hi"] < 0.01]
        nstar.append(dict(year=y, track=trk, Nstar_pairs=ns, Nstar_scenes=(2 * ns if ns else None),
                          first_M_below=(int(first["M"].iloc[0]) if len(first) else None),
                          rule="smallest M from which the upper 95% bound of dC stays < 1 pp for all larger M"))
        c2 = curve[(curve.year == y) & (curve.track == trk)]
        o2 = c2[c2.obs == "O2"].sort_values("M")
        o3 = c2[c2.obs == "O3"].sort_values("M")
        if len(o2) >= 3 and len(o3):
            xs, ys = o2["M"].values.astype(float), o2["BA"].values
            for _, r in o3.iterrows():
                target = r["BA"]
                if target > ys.max():
                    m2 = np.nan
                    flag = "above O2 curve (censored > 9)"
                else:
                    m2 = float(np.interp(target, np.maximum.accumulate(ys), xs))
                    flag = ""
                equiv.append(dict(year=y, track=trk, M=r["M"], BA_O3=target, M_equiv_O2=m2,
                                  multiplier=(m2 / r["M"] if np.isfinite(m2) else np.nan), note=flag))
        for m3, m2 in ((2, 3), (4, 6), (6, 9)):
            q3 = S.find(exp="E1", year_train=y, tracks=trk, M=m3, obs="O3", learner="HGB", variant="primary")
            q2 = S.find(exp="E1", year_train=y, tracks=trk, M=m2, obs="O2", learner="HGB", variant="primary")
            if len(q3) and len(q2):
                g3, g2 = S.get(q3.iloc[0].config_id, B=a.B_head), S.get(q2.iloc[0].config_id, B=a.B_head)
                d = g3["boot"] - g2["boot"]
                eqcost.append(dict(year=y, track=trk, credits=45 * m3, O3_M=m3, O2_M=m2,
                                   BA_O3=g3["point"], BA_O2=g2["point"], diff=g3["point"] - g2["point"],
                                   lo=ci(d)[0], hi=ci(d)[1], p=pval(d)))
        for obs in ("O2", "O3"):
            cc = c2[c2.obs == obs].sort_values("M")
            if len(cc) >= 4:
                try:
                    pr, _ = curve_fit(sat, cc["M"].values.astype(float), cc["BA"].values,
                                      p0=[cc["BA"].max(), cc["BA"].min(), 2.0], maxfev=20000)
                    fits.append(dict(year=y, track=trk, obs=obs, BA_inf=pr[0], BA_1=pr[1], kappa=pr[2]))
                except Exception as e:  # noqa: BLE001
                    fits.append(dict(year=y, track=trk, obs=obs, error=str(e)))
    for name, rows in (("T_Nstar", nstar), ("T_equivalent_scenes", equiv), ("T_equal_cost", eqcost), ("T_saturation_fit", fits)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)

    # ---- per-class gains
    pc = []
    for y in (2025, 2024):
        for trk in TRACKS:
            for M in BUDGETS:
                q2 = S.find(exp="E1", year_train=y, tracks=trk, M=M, obs="O2", learner="HGB", variant="primary")
                q3 = S.find(exp="E1", year_train=y, tracks=trk, M=M, obs="O3", learner="HGB", variant="primary")
                if not (len(q2) and len(q3)):
                    continue
                g2, g3 = S.get(q2.iloc[0].config_id), S.get(q3.iloc[0].config_id)
                r2, f2 = f1_from(g2["cm"].sum(0))
                r3, f3 = f1_from(g3["cm"].sum(0))
                _, bf2 = f1_from(g2["cmw"])
                _, bf3 = f1_from(g3["cmw"])
                for c, name in enumerate(TARGET_CLASSES):
                    d = bf3[:, c] - bf2[:, c]
                    pc.append(dict(year=y, track=trk, M=M, cls=name, recall_O2=r2[c], recall_O3=r3[c], F1_O2=f2[c],
                                   F1_O3=f3[c], dF1=f3[c] - f2[c], dF1_lo=ci(d)[0], dF1_hi=ci(d)[1]))
    pd.DataFrame(pc).to_csv(out / "T_E1_perclass.csv", index=False)

    # ---- E3 random draws
    e3 = []
    for (y, trk, M), g in S.find(exp="E3").groupby(["year_train", "tracks", "M"]) if len(S.find(exp="E3")) else []:
        vals = {}
        for obs in ("O2", "O3"):
            gg = g[g.obs == obs].sort_values("draw")
            vals[obs] = {r.draw: S.get(r.config_id) for r in gg.itertuples()}
        common = sorted(set(vals["O2"]) & set(vals["O3"]))
        if not common:
            continue
        ba2 = np.array([vals["O2"][d]["point"] for d in common])
        ba3 = np.array([vals["O3"][d]["point"] for d in common])
        boot_d = np.mean([vals["O3"][d]["boot"] - vals["O2"][d]["boot"] for d in common], axis=0)
        uni = {o: S.find(exp="E1", year_train=y, tracks=trk, M=M, obs=o, learner="HGB", variant="primary") for o in ("O2", "O3")}
        e3.append(dict(year=y, track=trk, M=M, n_draws=len(common), O2_mean=ba2.mean(), O2_sd=ba2.std(ddof=1),
                       O2_p5=np.percentile(ba2, 5), O2_p95=np.percentile(ba2, 95), O3_mean=ba3.mean(),
                       dC_mean=(ba3 - ba2).mean(), dC_lo=ci(boot_d)[0], dC_hi=ci(boot_d)[1],
                       O2_uniform=(S.get(uni["O2"].iloc[0].config_id)["point"] if len(uni["O2"]) else np.nan),
                       O3_uniform=(S.get(uni["O3"].iloc[0].config_id)["point"] if len(uni["O3"]) else np.nan),
                       draw_spread_O2=np.percentile(ba2, 95) - np.percentile(ba2, 5)))
    pd.DataFrame(e3).to_csv(out / "T_E3_random_draws.csv", index=False)

    # ---- E5 two tracks vs one
    e5 = []
    for r in S.find(exp="E5").itertuples():
        g = S.get(r.config_id)
        row = dict(year=r.year_train, M_total=r.M, obs=r.obs, BA_two=g["point"])
        for trk in TRACKS:
            q = S.find(exp="E1", year_train=r.year_train, tracks=trk, M=r.M, obs=r.obs, learner="HGB", variant="primary")
            if len(q):
                g1 = S.get(q.iloc[0].config_id)
                d = g["boot"] - g1["boot"]
                row.update({f"BA_T{trk}": g1["point"], f"diff_vs_T{trk}": g["point"] - g1["point"],
                            f"lo_T{trk}": ci(d)[0], f"hi_T{trk}": ci(d)[1]})
        e5.append(row)
    pd.DataFrame(e5).to_csv(out / "T_E5_two_tracks.csv", index=False)

    # ---- E9 cross-year
    e9 = []
    for (ya, yb, trk, M), g in S.find(exp="E9").groupby(["year_train", "year_test", "tracks", "M"]) if len(S.find(exp="E9")) else []:
        v = {r.obs: S.get(r.config_id) for r in g.itertuples()}
        if not {"O2", "O3"} <= set(v):
            continue
        d = v["O3"]["boot"] - v["O2"]["boot"]
        w = {o: S.find(exp="E1", year_train=yb, tracks=trk, M=M, obs=o, learner="HGB", variant="primary") for o in ("O2", "O3")}
        e9.append(dict(train=ya, test=yb, track=trk, M=M, BA_O2=v["O2"]["point"], BA_O3=v["O3"]["point"],
                       dC=v["O3"]["point"] - v["O2"]["point"], lo=ci(d)[0], hi=ci(d)[1],
                       within_O2=(S.get(w["O2"].iloc[0].config_id)["point"] if len(w["O2"]) else np.nan),
                       within_O3=(S.get(w["O3"].iloc[0].config_id)["point"] if len(w["O3"]) else np.nan)))
    pd.DataFrame(e9).to_csv(out / "T_E9_cross_year.csv", index=False)

    # ---- E10 robustness (+ pure>=4 subset of E1)
    e10 = []
    for (y, trk, M, lrn, var), g in S.find(exp="E10").groupby(["year_train", "tracks", "M", "learner", "variant"]) if len(S.find(exp="E10")) else []:
        v = {r.obs: S.get(r.config_id) for r in g.itertuples()}
        if {"O2", "O3"} <= set(v):
            d = v["O3"]["boot"] - v["O2"]["boot"]
            e10.append(dict(year=y, track=trk, M=M, learner=lrn, variant=var, BA_O2=v["O2"]["point"],
                            BA_O3=v["O3"]["point"], dC=v["O3"]["point"] - v["O2"]["point"], lo=ci(d)[0], hi=ci(d)[1]))
    for y in (2025, 2024):
        for trk in TRACKS:
            for M in BUDGETS:
                q2 = S.find(exp="E1", year_train=y, tracks=trk, M=M, obs="O2", learner="HGB", variant="primary")
                q3 = S.find(exp="E1", year_train=y, tracks=trk, M=M, obs="O3", learner="HGB", variant="primary")
                if len(q2) and len(q3):
                    g2, g3 = S.get(q2.iloc[0].config_id, subset="pure4"), S.get(q3.iloc[0].config_id, subset="pure4")
                    d = g3["boot"] - g2["boot"]
                    e10.append(dict(year=y, track=trk, M=M, learner="HGB", variant="pure_px>=4", BA_O2=g2["point"],
                                    BA_O3=g3["point"], dC=g3["point"] - g2["point"], lo=ci(d)[0], hi=ci(d)[1]))
    pd.DataFrame(e10).to_csv(out / "T_E10_robustness.csv", index=False)

    # ---- E11 controls
    e11 = []
    q = S.find(exp="E11")
    for (trk, obs, ctrl), g in q.groupby(["tracks", "obs", "control"]) if len(q) else []:
        pts = [S.get(r.config_id)["point"] for r in g.itertuples()]
        ref = S.find(exp="E1", year_train=2025, tracks=trk, M=9, obs=obs, learner="HGB", variant="primary") if trk != "37vs88" else []
        e11.append(dict(track=trk, obs=obs, control=ctrl, n=len(pts), BA_mean=float(np.mean(pts)), BA_max=float(np.max(pts)),
                        BA_true=(S.get(ref.iloc[0].config_id)["point"] if len(ref) else np.nan),
                        chance=1 / len(TARGET_CLASSES) if ctrl != "trackprobe" else 0.5))
    pd.DataFrame(e11).to_csv(out / "T_E11_controls.csv", index=False)

    # ---- E2w windows
    e2 = []
    q = S.find(exp="E2w")
    for w, g in q.groupby("window") if len(q) else []:
        v = {r.obs: S.get(r.config_id) for r in g.itertuples()}
        row = dict(window=w, **{f"BA_{k}": v[k]["point"] for k in v})
        for a_, b_ in (("W1_c12", "W0"), ("W5_c12_next", "W1_c12"), ("W4_all", "W1_c12"), ("W2_long", "W1_c12")):
            if a_ in v and b_ in v:
                d = v[a_]["boot"] - v[b_]["boot"]
                row.update({f"{a_}-{b_}": v[a_]["point"] - v[b_]["point"], f"{a_}-{b_}_lo": ci(d)[0],
                            f"{a_}-{b_}_hi": ci(d)[1]})
        e2.append(row)
    pd.DataFrame(e2).to_csv(out / "T_E2w_windows.csv", index=False)

    # ---- hypotheses
    H = {}
    d25 = dcs[dcs.year == 2025] if len(dcs) else pd.DataFrame()
    if len(d25):
        m1 = d25[d25.M == 1]
        H["H1"] = dict(dC_M1={int(r.track): [r.dC, r.lo, r.hi] for r in m1.itertuples()},
                       spearman_M_vs_dC={int(t): float(pd.Series(g.dC.values).corr(pd.Series(g.M.values), method="spearman"))
                                          for t, g in d25.groupby("track")},
                       falsified=bool((m1.hi < 0.03).all()))
        m9 = d25[d25.M == 9]
        ns = pd.DataFrame(nstar)
        H["H2"] = dict(Nstar={int(r.track): r.Nstar_pairs for r in ns[ns.year == 2025].itertuples()},
                       falsified=bool((m9.lo > 0.01).any()))
        pcd = pd.DataFrame(pc)
        h3 = {}
        for trk in TRACKS:
            nsv = ns[(ns.year == 2025) & (ns.track == trk)]["Nstar_pairs"]
            Mref = int(nsv.iloc[0]) if len(nsv) and pd.notna(nsv.iloc[0]) else 9
            for cls in ("TempGrass", "Maize"):
                r = pcd[(pcd.year == 2025) & (pcd.track == trk) & (pcd.M == Mref) & (pcd.cls == cls)]
                if len(r):
                    h3[f"T{trk}_{cls}_M{Mref}"] = [float(r.dF1.iloc[0]), float(r.dF1_lo.iloc[0]), float(r.dF1_hi.iloc[0])]
        H["H3"] = dict(dF1=h3, falsified=not all(v[1] > 0 for v in h3.values()) if h3 else None)
    e2d = pd.DataFrame(e2)
    if len(e2d) and "W5_c12_next-W1_c12_lo" in e2d:
        t88 = e2d[e2d.window.str.startswith("2025_T88")]
        H["H4"] = dict(windows={rec["window"]: [rec.get("W5_c12_next-W1_c12"), rec.get("W5_c12_next-W1_c12_lo"),
                                                rec.get("W5_c12_next-W1_c12_hi")] for rec in t88.to_dict("records")},
                       lower_bounds=t88["W5_c12_next-W1_c12_lo"].tolist(),
                       falsified=bool(not (t88["W5_c12_next-W1_c12_lo"] > 0).all()) if len(t88) else None)
    if len(nstar):
        ns = pd.DataFrame(nstar).fillna(10)
        h5 = {}
        for trk in TRACKS:
            a25 = ns[(ns.year == 2025) & (ns.track == trk)]["Nstar_pairs"]
            a24 = ns[(ns.year == 2024) & (ns.track == trk)]["Nstar_pairs"]
            if len(a25) and len(a24):
                h5[f"T{trk}"] = [float(a25.iloc[0]), float(a24.iloc[0])]
        H["H5"] = dict(Nstar_2025_2024=h5, falsified=any(abs(v[0] - v[1]) > 2 for v in h5.values()) if h5 else None,
                       note="undefined N* (no budget reached) coded as 10")
    write_json(out / "HYPOTHESES.json", {"utc": now(), **H})
    update_status("s7_stats", "DONE", configs=len(S.reg))
    log.info("stats written to %s", out)


if __name__ == "__main__":
    main()
