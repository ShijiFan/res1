"""R2 checker. The agent runs this; it decides PASS/FAIL, the agent does not.

  & $py a1_r2_check.py --before   # BEFORE anything else: fingerprints every protected file
  & $py a1_r2_check.py --after    # AFTER stats and figures: verifies nothing protected changed,
                                  # and that the R2 outputs contain the expected numbers

Writes reports/R2_PROTECT_BEFORE.json (--before) and reports/R2_CHECK.txt (--after).
Exit code 0 = all PASS, 1 = at least one FAIL. On FAIL: STOP and report R2_CHECK.txt verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import io
import sys
from pathlib import Path

import pandas as pd

from a1_common import D

RUN = D["stats"].parent
BEFORE = RUN / "reports" / "R2_PROTECT_BEFORE.json"
REPORT = RUN / "reports" / "R2_CHECK.txt"
PROTECTED_DIRS = ["stats", "figures", "manuscript"]
PROTECTED_FILES = ["reports/A1_PREREG.md", "reports/A1_PREREG.sha256"]

# Expected values, copied from stats/T_E3_random_draws.csv and stats/T_E1_deltaC.csv of the 2026-09-24 run.
# Units: percentage points of balanced accuracy, rounded to 0.01. Key: (year, track) -> {M: value}.
EXP_DC = {
    (2025, 37): {1: 1.23, 2: 1.49, 3: 0.80, 4: 0.68, 6: 0.24, 8: 0.28, 9: 0.39},
    (2025, 88): {1: 0.90, 2: 1.24, 3: 0.81, 4: 0.44, 6: 0.20, 8: 0.12, 9: 0.31},
    (2024, 37): {1: 0.99, 2: 1.67, 3: 1.61, 4: 1.60, 6: 0.84, 8: 0.47, 9: 0.37},
    (2024, 88): {1: 0.97, 2: 1.98, 3: 1.13, 4: 1.18, 6: 0.64, 8: 0.48, 9: 0.39},
}
EXP_O2 = {
    (2025, 37): {1: 45.78, 2: 63.03, 3: 73.21, 4: 77.58, 6: 82.32, 8: 83.86},
    (2025, 88): {1: 45.86, 2: 62.26, 3: 71.01, 4: 77.91, 6: 82.04, 8: 83.87},
    (2024, 37): {1: 38.28, 2: 51.85, 3: 63.08, 4: 65.23, 6: 72.65, 8: 75.62},
    (2024, 88): {1: 36.65, 2: 49.44, 3: 60.85, 4: 65.60, 6: 70.05, 8: 73.31},
}
# equal cost: O3 at M pairs minus O2 at 1.5 M pairs, key (year, track, O3_M) -> pp
EXP_EC = {
    (2025, 37, 2): -8.69, (2025, 37, 4): -4.05, (2025, 37, 6): -1.79,
    (2025, 88, 2): -7.51, (2025, 88, 4): -3.68, (2025, 88, 6): -2.24,
    (2024, 37, 2): -9.57, (2024, 37, 4): -5.81, (2024, 37, 6): -3.18,
    (2024, 88, 2): -9.43, (2024, 88, 4): -3.28, (2024, 88, 6): -3.49,
}
EXP_DRAWS = {2025: {1: 9, 2: 20, 3: 20, 4: 20, 6: 20, 8: 9, 9: 1}, 2024: {1: 9, 2: 10, 3: 10, 4: 10, 6: 10, 8: 9, 9: 1}}
FIGS = ["FigR2_1_budget_curves", "FigR2_2_coherence_gain", "FigR2_3_equal_cost", "FigR2_4_two_tracks"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint() -> dict:
    fp = {}
    for d in PROTECTED_DIRS:
        for p in sorted((RUN / d).rglob("*")):
            if p.is_file():
                fp[str(p.relative_to(RUN)).replace("\\", "/")] = sha(p)
    for f in PROTECTED_FILES:
        p = RUN / f
        fp[f] = sha(p) if p.exists() else "ABSENT"
    npz = sorted((RUN / "experiments").rglob("*.npz"))
    fp["__npz__"] = {str(p.relative_to(RUN)).replace("\\", "/"): p.stat().st_size for p in npz}
    reg = RUN / "experiments" / "REGISTRY.csv"
    fp["__registry_bytes__"] = reg.read_bytes().decode("utf-8")
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--before", action="store_true")
    g.add_argument("--after", action="store_true")
    a = ap.parse_args()

    if a.before:
        if BEFORE.exists():
            sys.exit(f"STOP: {BEFORE} already exists. Do not overwrite it. Report this to the user.")
        fp = fingerprint()
        BEFORE.write_text(json.dumps(fp), encoding="utf-8")
        n_reg = len(pd.read_csv(RUN / "experiments" / "REGISTRY.csv"))
        print(f"fingerprinted {len(fp) - 2} protected files, {len(fp['__npz__'])} npz, REGISTRY rows = {n_reg}")
        return

    lines, bad = [], 0

    def check(name, ok, detail=""):
        nonlocal bad
        bad += (not ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    old = json.loads(BEFORE.read_text(encoding="utf-8"))
    new = fingerprint()
    # 1. protected files unchanged
    for k, v in old.items():
        if k.startswith("__"):
            continue
        check(f"unchanged {k}", new.get(k) == v)
    extra = [k for k in new if not k.startswith("__") and k not in old]
    check("no new files inside stats/ figures/ manuscript/", not extra, ", ".join(extra[:5]))
    # 2. old npz untouched
    changed = [k for k, s in old["__npz__"].items() if new["__npz__"].get(k) != s]
    check("all pre-existing npz files present with same size", not changed, ", ".join(changed[:5]))
    added = [k for k in new["__npz__"] if k not in old["__npz__"]]
    check("new npz only under experiments/E5r/", all(k.startswith("experiments/E5r/") for k in added),
          f"{len(added)} new npz")
    # 3. registry: old content is an exact prefix, only E5r rows appended
    check("REGISTRY.csv old rows byte-identical (prefix)", new["__registry_bytes__"].startswith(old["__registry_bytes__"]))
    reg = pd.read_csv(RUN / "experiments" / "REGISTRY.csv")
    n_old = len(pd.read_csv(io.StringIO(old["__registry_bytes__"])))
    app = reg.iloc[n_old:]
    check("appended rows are all exp=E5r", set(app.exp) <= {"E5r"}, f"{len(app)} appended")
    check("E5r rows = 240", (reg.exp == "E5r").sum() == 240, str((reg.exp == "E5r").sum()))
    check("no duplicate config_id", not reg.config_id.duplicated().any())
    e5 = reg[reg.exp == "E5r"]
    if len(e5):
        per = e5.groupby(["year_train", "M"]).size().to_dict()
        want = {(y, m): 2 * n for y, n in ((2025, 20), (2024, 10)) for m in (2, 4, 6, 8)}
        check("E5r rows per (year, M_total) = 40 (2025) / 20 (2024)", per == want, str(per))
        check("E5r variant = exploratory_r2", set(e5.variant) == {"exploratory_r2"})
        check("E5r 120 distinct designs", e5.pairs.nunique() == 120, str(e5.pairs.nunique()))
    # 4. R2_RUN.json
    rr = json.loads((RUN / "stats_r2" / "R2_RUN.json").read_text(encoding="utf-8"))
    check("R2_RUN final = true", rr.get("final") is True)
    check("R2_RUN allow_missing = false", rr.get("allow_missing") is False)
    check("R2_RUN gates = G1 G2 G3 G4", rr.get("gates_passed") == ["G1", "G2", "G3", "G4"], str(rr.get("gates_passed")))
    check("R2_RUN missing files = 0", rr.get("n_missing_files") == 0)
    # 5. numbers
    cur = pd.read_csv(RUN / "stats_r2" / "T_R2A_E3_curve.csv")
    check("T_R2A has 28 rows", len(cur) == 28, str(len(cur)))
    for (y, t), row in EXP_DC.items():
        for M, v in row.items():
            r = cur[(cur.year == y) & (cur.track == t) & (cur.M == M)]
            got = round(100 * float(r.dC.iloc[0]), 2) if len(r) else None
            check(f"dC {y} T{t} M={M} = {v:.2f} pp", got is not None and abs(got - v) <= 0.011, f"got {got}")
            if len(r):
                check(f"n_draws {y} T{t} M={M} = {EXP_DRAWS[y][M]}", int(r.n_draws.iloc[0]) == EXP_DRAWS[y][M],
                      f"got {int(r.n_draws.iloc[0])}")
                check(f"CI brackets dC {y} T{t} M={M}", r.dC_lo.iloc[0] <= r.dC.iloc[0] <= r.dC_hi.iloc[0])
    for (y, t), row in EXP_O2.items():
        for M, v in row.items():
            r = cur[(cur.year == y) & (cur.track == t) & (cur.M == M)]
            got = round(100 * float(r.BA_O2.iloc[0]), 2) if len(r) else None
            check(f"BA_O2 {y} T{t} M={M} = {v:.2f} %", got is not None and abs(got - v) <= 0.011, f"got {got}")
    ec = pd.read_csv(RUN / "stats_r2" / "T_R2D_equal_cost_E3.csv")
    check("T_R2D has 12 rows", len(ec) == 12, str(len(ec)))
    for (y, t, m), v in EXP_EC.items():
        r = ec[(ec.year == y) & (ec.track == t) & (ec.O3_M == m)]
        got = round(100 * float(r["diff"].iloc[0]), 2) if len(r) else None
        check(f"equal-cost {y} T{t} O3 M={m} = {v:.2f} pp", got is not None and abs(got - v) <= 0.011, f"got {got}")
    for bad_str in ("p < 0.0001", "p < 1e-04"):
        check(f"no '{bad_str}' in any R2 table",
              not any(bad_str in (RUN / "stats_r2" / f).read_text(encoding="utf-8")
                      for f in ("T_R2A_E3_curve.csv", "T_R2D_equal_cost_E3.csv", "T_R2E_two_tracks_E5r.csv")))
    q6 = pd.read_csv(RUN / "stats_r2" / "T_R2E_two_tracks_E5r.csv")
    check("T_R2E has 16 rows (2 years x 4 M_total x 2 obs)", len(q6) == 16, str(len(q6)))
    h5 = json.loads((RUN / "stats_r2" / "H5_R2.json").read_text(encoding="utf-8"))
    check("H5_R2 overall status present", h5.get("overall") in ("CONFIRMED", "FALSIFIED", "NOT_EVALUABLE"),
          str(h5.get("overall")))
    # 6. figures
    for f in FIGS:
        for ext in ("pdf", "png"):
            p = RUN / "figures_r2" / f"{f}.{ext}"
            check(f"figure {f}.{ext} exists", p.exists() and p.stat().st_size > 5000)

    lines.insert(0, f"R2 CHECK: {'ALL PASS' if bad == 0 else f'{bad} FAIL'}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join([lines[0]] + [l for l in lines[1:] if l.startswith("FAIL")]))
    print(f"full report: {REPORT}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
