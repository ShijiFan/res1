"""Site 2, Study 2: apply the five pre-registered replication criteria R2-1..R2-5 (PREREG_SITE2.md section 3)
mechanically to the statistics tables. Written before the site-2 statistics existed.

R2-1  E3, M <= 2: dC lower 95% bound > 0 in >= 6 of 8 cells (2 years x 2 tracks x M in {1, 2})
R2-2  E3, M = 8: dC point estimate < 1 pp in all 4 year-track cells
R2-3  equal cost, 90 and 180 credits: upper bound of O3@M - O2@1.5M < 0 in >= 6 of 8 cells
R2-4  uniform design: dP point > dC point in >= 90% of year-track-budget cells
R2-5  learner: dC(LR) > dC(RF) in >= 12 of 16 cells (E10, primary variant)
Out: ../RESULTS_SITE2_STUDY2_CRITERIA.json (+ printed table)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "MAIN"


def main():
    e3 = pd.read_csv(RUN / "stats_r2" / "T_R2A_E3_curve.csv")
    ec = pd.read_csv(RUN / "stats_r2" / "T_R2D_equal_cost_E3.csv")
    e1 = pd.read_csv(RUN / "stats" / "T_E1_deltaC.csv")
    e10 = pd.read_csv(RUN / "stats" / "T_E10_robustness.csv")
    e10 = e10[e10.variant == "primary"]
    out = {}
    c1 = e3[e3.M.isin([1, 2])]
    n1 = int((c1.dC_lo > 0).sum())
    out["R2-1"] = {"cells": len(c1), "lower_bound_gt0": n1, "replicated": n1 >= 6,
                   "values_pp": (c1.dC * 100).round(2).tolist()}
    c2 = e3[e3.M == 8]
    out["R2-2"] = {"cells": len(c2), "point_lt_1pp": int((c2.dC < 0.01).sum()),
                   "replicated": bool((c2.dC < 0.01).all()), "values_pp": (c2.dC * 100).round(2).tolist()}
    c3 = ec[ec.credits.isin([90, 180])]
    n3 = int((c3.hi < 0).sum())
    out["R2-3"] = {"cells": len(c3), "upper_bound_lt0": n3, "replicated": n3 >= 6,
                   "values_pp": (c3["diff"] * 100).round(2).tolist()}
    n4 = int((e1.dPol > e1.dC).sum())
    out["R2-4"] = {"cells": len(e1), "dP_gt_dC": n4, "replicated": n4 >= 0.9 * len(e1)}
    piv = e10.pivot_table(index=["year", "track", "M"], columns="learner", values="dC")
    n5 = int((piv["LR"] > piv["RF"]).sum())
    out["R2-5"] = {"cells": len(piv), "LR_gt_RF": n5, "replicated": n5 >= 12}
    (ROOT / "RESULTS_SITE2_STUDY2_CRITERIA.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k, "REPLICATED" if v["replicated"] else "NOT REPLICATED", {a: b for a, b in v.items() if a != "replicated"})


if __name__ == "__main__":
    main()
