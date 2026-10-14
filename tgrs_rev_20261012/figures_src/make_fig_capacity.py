"""Fig. 9 (coherence increment by learner and CNN patch size).

Same inputs and values as merged_tgrs_20260930/plan/substantive_20261002/make_revision_figures.py (capacity
part); the only change is presentation: one marker style for all patch sizes (the CPU/CUDA backend note moved
to the supplementary material).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402  (OUT -> this revision's manuscript/figures)

R = HERE.parents[1]
SUB = R / "merged_tgrs_20260930" / "plan" / "substantive_20261002"
S = R / "merged_tgrs_20260930" / "figures_src"

fig = plt.figure(figsize=(6.8, 2.9))
axs = fig.subplots(1, 2)
tab = pd.read_csv(R / "A0_gamma0_rerun_20260929/ws/tables/T1_contrasts.csv")
tb = pd.read_csv(R / "site2_20260930/runs/STUDY1/ws/tables/R1_contrasts.csv")
cc = pd.read_csv(R / "A0_gamma0_rerun_20260929/ws/tables_cnn/contrasts_checks.csv")
cb = json.loads((S / "site2_cnn_contrast.json").read_text())
for site, offset, color in [("A", -.1, "#0072B2"), ("B", .1, "#D55E00")]:
    vals = []
    for learner in ["RF", "SVC"]:
        if site == "A":
            row = tab[(tab.contrast_id == "C-main") & (tab.learner == learner) & (tab.metric == "cross_ba")].iloc[0]
            vals.append([row.point_estimate, row.ci_lo, row.ci_hi])
        else:
            row = tb[(tb.contrast == "C-main") & (tb.learner == learner) & (tb.metric == "cross_ba")].iloc[0]
            vals.append([row.delta, row.ci_lo, row.ci_hi])
    cr = cc[(cc.family == "tuned_p9") & (cc["eval"] == "sample") & (cc.contrast == "CNN_IG - CNN_I")].iloc[0]
    vals.append([cr.delta_cross_ba, cr.ci_lo, cr.ci_hi] if site == "A" else [cb["delta"], cb["ci_lo"], cb["ci_hi"]])
    v = np.array(vals)
    axs[0].errorbar(np.arange(3) + offset, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o",
                    color=color, label=f"Site {site}", ms=4)
axs[0].set_xticks(range(3), ["RF", "SVC", "CNN 9×9"])
axs[0].set_title("(a) Learner", loc="left")
axs[0].set_ylabel("Coherence increment (pp BA)")
axs[0].legend(frameon=False)
t = pd.read_csv(SUB / "cnn/contrasts.csv")
t = t[t["eval"] == "sample"].copy()
bc = json.loads((SUB / "cnn/backend_control.json").read_text())
t.loc[t.patch == 5, ["delta", "lo", "hi"]] = [bc["delta"], *bc["ci"]]
t = t.sort_values("patch")
b640 = pd.read_csv(HERE.parent / "cnn640/contrasts_buffers.csv")
b640 = b640[(b640.series == "B640") & (b640["eval"] == "sample")].sort_values("patch")
xs = np.arange(len(t))
for d, off, color, mk, lab in [(t, -.08, "#0072B2", "o", "400 m exclusion"),
                               (b640, .08, "#009E73", "s", "640 m exclusion")]:
    axs[1].plot(xs + off, d.delta, color=color, lw=1)
    axs[1].errorbar(xs + off, d.delta, yerr=[d.delta - d.lo, d.hi - d.delta], fmt=mk, color=color, ms=4,
                    label=lab)
axs[1].legend(frameon=False)
axs[1].set_xticks(xs, [f"{p}×{p}" for p in t.patch])
axs[1].set_title("(b) Spatial context", loc="left")
axs[1].set_xlabel("CNN patch (Site A, fixed training)")
for ax in axs:
    ax.axhline(0, color=".45", lw=.6)
    ax.set_ylim(-2, 12)
    mf.grid(ax, "y")
fig.tight_layout()
print(t[["patch", "delta", "lo", "hi"]].round(2).to_string(index=False))
mf.save(fig, "fig_capacity")
