"""R2 statistics: corrected analyses after the 2026-09-24 review (A1_MAIN_RUN_REVIEW_20260924.md).

READ-ONLY on everything produced by a1_s6 / a1_s7. Writes only to RUN/stats_r2/.

What this script adds (all labelled EXPLORATORY in the output, except R2-A which re-reports
preregistered E3 data with a correct interval):

  R2-A  E3 budget curve: mean over random date draws, with a two-level bootstrap
        (5 km blocks  x  date draws). Replaces nothing; it is the date-choice-robust summary.
  R2-B  Where the preregistered uniform design sits inside the E3 draw distribution.
  R2-C  N* recomputed on the E3 curve, and H5 re-evaluated with an explicit status
        (CONFIRMED / FALSIFIED / NOT_EVALUABLE). No imputation of undefined N*.
  R2-D  Equal-cost contrast on E3 draws: O3 at M pairs vs O2 at 1.5 M pairs.
  R2-E  Two tracks vs one track (Q6) on random draws, if experiments/E5r exists
        (produced by a1_r2_e5r.py).
  R2-F  p-value strings that respect the bootstrap resolution (never "p < 0.0001" when B = 2000).

Usage (PowerShell):
  & $py a1_r2_stats.py                 # full run, every required file must exist
  & $py a1_r2_stats.py --allow-missing # ONLY for testing on a partial copy; never for final numbers
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np
import pandas as pd

from a1_common import D, TRACKS, get_logger, now, write_json
from a1_s7_stats import Store, ci, pval

log = get_logger("r2_stats")
OUT = D["stats"].parent / "stats_r2"
N_AVAIL_PAIRS = 9          # every year x track has exactly 9 disjoint 12-day pairs (E1 M=9)
SEED_R2 = 20260925


# ------------------------------------------------------------------ helpers
def fmt_p(p: float, B: int) -> str:
    """Bootstrap p-values cannot be smaller than 1/B. Report that bound instead of 0."""
    if not np.isfinite(p):
        return "n/a"
    floor = 1.0 / B
    if p < floor:
        return f"p < {floor:.1g}"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}"


def two_level(points: np.ndarray, boots: np.ndarray, exhaustive: bool, rng: np.random.Generator):
    """points: (K,) point BA (or difference) per draw; boots: (K, B) block-bootstrap replicates per draw,
    computed with IDENTICAL block weights for every draw (Store guarantees this within a year).
    Returns (point_mean, replicate_means (B,)).
    exhaustive=True means the K draws are the complete set of designs (e.g. all 9 single pairs); then
    the mean over designs is a fixed target and only the block level is resampled."""
    K, B = boots.shape
    if exhaustive or K == 1:
        return float(points.mean()), boots.mean(axis=0)
    idx = rng.integers(0, K, size=(B, K))
    rep = boots[idx, np.arange(B)[:, None]].mean(axis=1)
    return float(points.mean()), rep


class R2:
    def __init__(self, B: int, allow_missing: bool):
        self.S = Store(B, B)
        self.B = B
        self.allow_missing = allow_missing
        self.missing = []
        reg = self.S.reg
        self.reg = reg

    def _get(self, cid):
        r = self.reg[self.reg.config_id == cid].iloc[0]
        p = D["exp"] / r["path"]
        if not p.exists():
            if self.allow_missing:
                self.missing.append(str(p))
                return None
            raise FileNotFoundError(f"STOP: required prediction file missing: {p}")
        return self.S.get(cid, B=self.B)

    def draws(self, exp, year, track, M, obs, tracks_field=None):
        """All draws of one design cell. Returns dict draw -> result."""
        q = self.S.find(exp=exp, year_train=year, tracks=(tracks_field or track), M=M, obs=obs, learner="HGB")
        out = {}
        for r in q.itertuples():
            g = self._get(r.config_id)
            if g is not None:
                out[str(r.draw)] = dict(g=g, pairs=r.pairs)
        return out

    def uniform(self, year, track, M, obs):
        q = self.S.find(exp="E1", year_train=year, tracks=track, M=M, obs=obs, learner="HGB", variant="primary")
        return self._get(q.iloc[0].config_id) if len(q) else None


def cell(R: R2, year, track, M, rng):
    """E3 cell (or E1 M=9, the only design with 9 pairs)."""
    if M == 9:
        g2, g3 = R.uniform(year, track, 9, "O2"), R.uniform(year, track, 9, "O3")
        if g2 is None or g3 is None:
            return None
        d2 = {"all": dict(g=g2, pairs="")}
        d3 = {"all": dict(g=g3, pairs="")}
    else:
        d2 = R.draws("E3", year, track, M, "O2")
        d3 = R.draws("E3", year, track, M, "O3")
    common = sorted(set(d2) & set(d3), key=lambda s: (len(s), s))
    if not common:
        return None
    exhaustive = (M == 9) or (len(common) == comb(N_AVAIL_PAIRS, M))
    p2 = np.array([d2[k]["g"]["point"] for k in common])
    p3 = np.array([d3[k]["g"]["point"] for k in common])
    b2 = np.vstack([d2[k]["g"]["boot"] for k in common])
    b3 = np.vstack([d3[k]["g"]["boot"] for k in common])
    # one shared draw-resampling index for O2, O3 and their difference (paired over draws as well)
    seed = int(rng.integers(0, 2**31 - 1))
    m2, r2 = two_level(p2, b2, exhaustive, np.random.default_rng(seed))
    m3, r3 = two_level(p3, b3, exhaustive, np.random.default_rng(seed))
    md, rd = two_level(p3 - p2, b3 - b2, exhaustive, np.random.default_rng(seed))
    return dict(K=len(common), exhaustive=exhaustive, p2=p2, p3=p3, m2=m2, r2=r2, m3=m3, r3=r3, md=md, rd=rd,
                b2=b2, b3=b3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R = R2(a.B, a.allow_missing)
    rng = np.random.default_rng(SEED_R2)
    budgets = (1, 2, 3, 4, 6, 8, 9)
    cells = {}

    # ---------------- R2-A / R2-B: E3 curve and the uniform design's rank
    rows_curve, rows_rank = [], []
    for y in (2025, 2024):
        for trk in TRACKS:
            for M in budgets:
                c = cell(R, y, trk, M, rng)
                if c is None:
                    continue
                cells[(y, trk, M)] = c
                lo2, hi2 = ci(c["r2"]); lo3, hi3 = ci(c["r3"]); lod, hid = ci(c["rd"])
                p = pval(c["rd"])
                rows_curve.append(dict(year=y, track=trk, M=M, N=2 * M, n_draws=c["K"], exhaustive=c["exhaustive"],
                                       BA_O2=c["m2"], BA_O2_lo=lo2, BA_O2_hi=hi2,
                                       BA_O3=c["m3"], BA_O3_lo=lo3, BA_O3_hi=hi3,
                                       dC=c["md"], dC_lo=lod, dC_hi=hid, p=p, p_str=fmt_p(p, a.B),
                                       status="PREREG_E3_REPORTED_WITH_TWO_LEVEL_CI"))
                u2 = R.uniform(y, trk, M, "O2")
                u3 = R.uniform(y, trk, M, "O3")
                if u2 is not None and u3 is not None and M < 9:
                    rows_rank.append(dict(year=y, track=trk, M=M, n_draws=c["K"],
                                          O2_uniform=u2["point"], O2_draw_mean=c["m2"],
                                          O2_draw_min=float(c["p2"].min()), O2_draw_p5=float(np.percentile(c["p2"], 5)),
                                          O2_draw_p95=float(np.percentile(c["p2"], 95)),
                                          uniform_rank_frac=float((c["p2"] < u2["point"]).mean()),
                                          dC_uniform=u3["point"] - u2["point"], dC_draw_mean=c["md"],
                                          status="EXPLORATORY"))
    curve = pd.DataFrame(rows_curve)
    curve.to_csv(OUT / "T_R2A_E3_curve.csv", index=False)

    # ---------------- GATES (only in a final run; --allow-missing skips them)
    if not a.allow_missing:
        # G1: every cell exists with the expected number of draws
        exp_n = {1: 9, 8: 9, 9: 1}
        for y in (2025, 2024):
            for trk in TRACKS:
                for M in budgets:
                    want = exp_n.get(M, 20 if y == 2025 else 10)
                    c = cells.get((y, trk, M))
                    assert c is not None, f"G1 FAIL: cell {y} T{trk} M={M} missing"
                    assert c["K"] == want, f"G1 FAIL: {y} T{trk} M={M} has {c['K']} draws, expected {want}"
        # G2: point means must reproduce the old s7 table exactly (same predictions, same metric)
        old = pd.read_csv(D["stats"] / "T_E3_random_draws.csv")
        for r in old.itertuples():
            c = cells[(int(r.year), int(r.track), int(r.M))]
            assert abs(c["m2"] - r.O2_mean) < 1e-9, f"G2 FAIL O2 {r.year} T{r.track} M={r.M}: {c['m2']} vs {r.O2_mean}"
            assert abs(c["m3"] - r.O3_mean) < 1e-9, f"G2 FAIL O3 {r.year} T{r.track} M={r.M}"
            assert abs(c["md"] - r.dC_mean) < 1e-9, f"G2 FAIL dC {r.year} T{r.track} M={r.M}"
        # G3: M=9 must reproduce the preregistered E1 value
        e1 = pd.read_csv(D["stats"] / "T_E1_deltaC.csv")
        for r in e1[e1.M == 9].itertuples():
            c = cells[(int(r.year), int(r.track), 9)]
            assert abs(c["md"] - r.dC) < 1e-9, f"G3 FAIL {r.year} T{r.track} M=9"
        log.info("gates G1-G3 passed")
        gates = ["G1", "G2", "G3"]
    else:
        gates = []
    pd.DataFrame(rows_rank).to_csv(OUT / "T_R2B_uniform_rank.csv", index=False)

    # ---------------- R2-C: N* on the E3 curve and H5 with an explicit status
    rows_ns = []
    for (y, trk), g in curve.groupby(["year", "track"]):
        g = g.sort_values("M")
        ok = (g["dC_hi"] < 0.01).values
        ns = None
        for i in range(len(g)):
            if ok[i:].all():
                ns = int(g["M"].iloc[i]); break
        rows_ns.append(dict(year=y, track=trk, Nstar_E3=ns, defined=ns is not None,
                            rule="smallest M from which the upper 95% bound of the E3 dC stays < 1 pp for all larger M",
                            status="EXPLORATORY"))
    ns_e3 = pd.DataFrame(rows_ns)
    ns_e3.to_csv(OUT / "T_R2C_Nstar_E3.csv", index=False)

    ns_pre = pd.read_csv(D["stats"] / "T_Nstar.csv")          # preregistered (uniform design) N*
    h5 = {}
    for trk in TRACKS:
        a25 = ns_pre[(ns_pre.year == 2025) & (ns_pre.track == trk)]["Nstar_pairs"]
        a24 = ns_pre[(ns_pre.year == 2024) & (ns_pre.track == trk)]["Nstar_pairs"]
        v25 = float(a25.iloc[0]) if len(a25) and pd.notna(a25.iloc[0]) else None
        v24 = float(a24.iloc[0]) if len(a24) and pd.notna(a24.iloc[0]) else None
        if v25 is None or v24 is None:
            st = "NOT_EVALUABLE"
        else:
            st = "FALSIFIED" if abs(v25 - v24) > 2 else "CONFIRMED"
        h5[f"T{trk}"] = dict(Nstar_2025=v25, Nstar_2024=v24, status=st)
    overall = ("NOT_EVALUABLE" if any(v["status"] == "NOT_EVALUABLE" for v in h5.values())
               else "FALSIFIED" if any(v["status"] == "FALSIFIED" for v in h5.values()) else "CONFIRMED")
    write_json(OUT / "H5_R2.json", dict(
        utc=now(), per_track=h5, overall=overall,
        note=("Preregistered H5 is |N*(2025) - N*(2024)| <= 2. The preregistration contains no rule for an "
              "undefined N*. a1_s7_stats.py coded undefined as 10 (fillna(10)); that coding is not preregistered "
              "and is NOT used here. If N* is undefined in either year the hypothesis cannot be evaluated.")))

    # ---------------- R2-D: equal cost on E3 draws
    rows_ec = []
    for y in (2025, 2024):
        for trk in TRACKS:
            for m3, m2 in ((2, 3), (4, 6), (6, 9)):
                c3, c2 = cells.get((y, trk, m3)), cells.get((y, trk, m2))
                if c3 is None or c2 is None:
                    continue
                s = int(rng.integers(0, 2**31 - 1))
                mA, rA = two_level(c3["p3"], c3["b3"], c3["exhaustive"], np.random.default_rng(s))
                mB, rB = two_level(c2["p2"], c2["b2"], c2["exhaustive"], np.random.default_rng(s + 1))
                d = rA - rB
                p = pval(d)
                rows_ec.append(dict(year=y, track=trk, credits=45 * m3, O3_M=m3, O2_M=m2, BA_O3_draws=mA,
                                    BA_O2_draws=mB, diff=mA - mB, lo=ci(d)[0], hi=ci(d)[1], p=p, p_str=fmt_p(p, a.B),
                                    price_assumption="RTC 20 m = 15 credits, INSAR 10x2 = 15 credits (HyP3 on-demand)",
                                    status="EXPLORATORY"))
    pd.DataFrame(rows_ec).to_csv(OUT / "T_R2D_equal_cost_E3.csv", index=False)

    # ---------------- R2-E: two tracks vs one on random draws (only if E5r was run)
    rows_q6 = []
    n_e5r = len(R.S.find(exp="E5r"))
    if n_e5r and not a.allow_missing:
        assert n_e5r == 240, f"G4 FAIL: E5r has {n_e5r} registry rows, expected 240"
        gates.append("G4")
    if n_e5r:
        for y in (2025, 2024):
            for Mt in (2, 4, 6, 8):
                for obs in ("O2", "O3"):
                    two = R.draws("E5r", y, None, Mt, obs, tracks_field="37+88")
                    if not two:
                        continue
                    pT = np.array([v["g"]["point"] for v in two.values()])
                    bT = np.vstack([v["g"]["boot"] for v in two.values()])
                    s = int(rng.integers(0, 2**31 - 1))
                    mT, rT = two_level(pT, bT, False, np.random.default_rng(s))
                    row = dict(year=y, M_total=Mt, obs=obs, n_draws_two=len(two), BA_two=mT)
                    for trk in TRACKS:
                        c = cells.get((y, trk, Mt))
                        if c is None:
                            continue
                        pS, bS = (c["p2"], c["b2"]) if obs == "O2" else (c["p3"], c["b3"])
                        mS, rS = two_level(pS, bS, c["exhaustive"], np.random.default_rng(s + trk))
                        d = rT - rS
                        p = pval(d)
                        row.update({f"BA_T{trk}": mS, f"diff_vs_T{trk}": mT - mS, f"lo_T{trk}": ci(d)[0],
                                    f"hi_T{trk}": ci(d)[1], f"p_T{trk}": fmt_p(p, a.B)})
                    row["status"] = "EXPLORATORY"
                    rows_q6.append(row)
    pd.DataFrame(rows_q6).to_csv(OUT / "T_R2E_two_tracks_E5r.csv", index=False)

    # ---------------- bookkeeping
    write_json(OUT / "R2_RUN.json", dict(utc=now(), B=a.B, allow_missing=a.allow_missing, gates_passed=gates,
                                         e5r_rows=n_e5r,
                                         n_missing_files=len(R.missing), missing_examples=R.missing[:20],
                                         final=(not a.allow_missing) and not R.missing))
    log.info("R2 stats written to %s (missing files: %d)", OUT, len(R.missing))


if __name__ == "__main__":
    main()
