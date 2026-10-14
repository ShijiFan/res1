"""Supplementary figure: Study 2 absolute BA of O2 and O3 and relative error reduction vs budget.

Input: stats_r2/T_R2A_E3_curve.csv at both sites (the file behind Fig. 11). RER = dC / (1 - BA_O2),
a descriptive point estimate per site-year-track cell (no interval).
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure_typography import normalize_figure

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
SRC = {"Site A": SAR / "A1_observation_budget_20260923/runs/20260924_A1_MAIN/stats_r2/T_R2A_E3_curve.csv",
       "Site B": SAR / "site2_20260930/runs/MAIN/stats_r2/T_R2A_E3_curve.csv"}
STY = {(37, 2025): ("#2a78d6", "-"), (37, 2024): ("#2a78d6", "--"), (88, 2025): ("#eb6834", "-"),
       (88, 2024): ("#eb6834", "--")}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
rows = []
for j, (site, p) in enumerate(SRC.items()):
    e = pd.read_csv(p)
    for (trk, yr), (c, ls) in STY.items():
        g = e[(e.track == trk) & (e.year == yr)].sort_values("M")
        lab = f"T{trk} {yr}"
        axs[0, j].plot(g.M, g.BA_O2 * 100, color=c, ls=ls, lw=1.1, marker="o", ms=2.5, label=f"{lab}, $O_2$")
        axs[0, j].plot(g.M, g.BA_O3 * 100, color=c, ls=ls, lw=0.6, marker="^", ms=2.5, alpha=0.7)
        rer = g.dC / (1 - g.BA_O2) * 100
        axs[1, j].plot(g.M, rer, color=c, ls=ls, lw=1.1, marker="o", ms=2.5, label=lab)
        for m, a, b, r in zip(g.M, g.BA_O2, g.BA_O3, rer):
            rows.append(dict(site=site, year=yr, track=trk, M=int(m), BA_O2=round(a * 100, 2),
                             BA_O3=round(b * 100, 2), RER_pct=round(r, 2)))
    axs[0, j].set_title(f"({'ab'[j]}) {site}: BA", loc="left")
    axs[1, j].set_title(f"({'cd'[j]}) {site}: relative error reduction", loc="left")
    axs[1, j].set_xlabel("Repeat pairs $M$")
    axs[1, j].axhline(0, color=".45", lw=0.6)
    for ax in axs[:, j]:
        ax.set_xticks([1, 2, 3, 4, 6, 8, 9])
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
axs[0, 0].set_ylabel("Balanced accuracy (%)")
axs[1, 0].set_ylabel("$\\Delta_C$ / (100 − BA$_{O_2}$) (%)")
h, l = axs[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.text(0.5, 0.955, "Thick lines with circles: $O_2$ (backscatter); thin lines with triangles: $O_3$ (with coherence)",
         ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
normalize_figure(fig)
for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
    fig.savefig(OUT / f"figS_budget_levels.{ext}", **kw)
df = pd.DataFrame(rows)
df.to_csv(HERE / "study2_levels_rer.csv", index=False)
(HERE / "figS_budget_levels_manifest.json").write_text(json.dumps(
    {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.values()}, indent=2))
s = df[df.M.isin([1, 2, 8, 9])].groupby(["site", "M"]).agg(BA_O2_min=("BA_O2", "min"), BA_O2_max=("BA_O2", "max"),
                                                           RER_min=("RER_pct", "min"), RER_max=("RER_pct", "max"))
print(s.to_string())
