"""R2 figures, drawn at final print size in the house style of E:\\research\\SAR\\paper\\figures\\tgrs_style.py
(DejaVu Sans 8 pt, Okabe-Ito colours, 0.6 pt axes, panel letters below the axes).
Reads RUN/stats_r2/*.csv and RUN/stats/*.csv only. Writes RUN/figures_r2/*.pdf and *.png.

  FigR2_1_budget_curves   BA vs scenes: random-date mean with 95% band (lines) vs the uniform design (hollow)
  FigR2_2_coherence_gain  dC vs scenes: random-date mean with 95% CI vs the uniform design
  FigR2_3_equal_cost      O3 at M pairs minus O2 at 1.5 M pairs, random dates, with 95% CI
  FigR2_4_two_tracks      only if T_R2E_two_tracks_E5r.csv has rows
"""
from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a1_common import D

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_O2, C_O3, C_T37, C_T88, C_GREY = "#D55E00", "#0072B2", "#0072B2", "#009E73", "#999999"
SINGLE, DOUBLE = 3.5, 7.16
IN = D["stats"].parent / "stats_r2"
OUT = D["stats"].parent / "figures_r2"


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color="#dddddd", lw=0.4)
    ax.set_axisbelow(True)


def letters(fig, axes, pad=0.012):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, L in zip(axes, "abcdefgh"):
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        p = ax.get_position()
        fig.text(p.x0 + p.width / 2, bb.y0 - pad, f"({L})", ha="center", va="top", fontsize=8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def fig1():
    cur = pd.read_csv(IN / "T_R2A_E3_curve.csv")
    uni = pd.read_csv(D["stats"] / "T_E1_curve.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.3), sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.55, wspace=0.08))
    for ax, (y, trk) in zip(axes.flat, [(2025, 37), (2025, 88), (2024, 37), (2024, 88)]):
        c = cur[(cur.year == y) & (cur.track == trk)].sort_values("M")
        for obs, col, lab in (("O2", C_O2, "dual-pol backscatter"), ("O3", C_O3, "backscatter + 12-day coherence")):
            ax.fill_between(c.N, 100 * c[f"BA_{obs}_lo"], 100 * c[f"BA_{obs}_hi"], color=col, alpha=0.18, lw=0)
            ax.plot(c.N, 100 * c[f"BA_{obs}"], color=col, lw=1.2, marker="o", ms=3.2, label=f"{lab}, random dates")
            u = uni[(uni.year == y) & (uni.track == trk) & (uni.obs == obs)].sort_values("M")
            ax.plot(u.N, 100 * u.BA, color=col, lw=0.7, ls=(0, (2.5, 1.5)), marker="o", ms=3.4, mfc="white",
                    mew=0.8, label=f"{lab}, uniform design")
        u2 = uni[(uni.year == y) & (uni.track == trk) & (uni.obs == "O2") & (uni.M == 2)]
        if len(u2):
            ax.annotate("Mar + Oct\nonly", xy=(4, 100 * u2.BA.iloc[0]), xytext=(8.6, 100 * u2.BA.iloc[0] + 0.5),
                        fontsize=6.2, color="#444444", arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888"))
        ax.set_title(f"{y}  T{trk}", fontsize=7.4, pad=3)
        ax.set_xticks([2, 4, 6, 8, 12, 16, 18])
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("balanced accuracy (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("scenes N (= 2 \u00d7 pairs M)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.86)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=2, fontsize=6.6)
    letters(fig, axes.flat)
    save(fig, "FigR2_1_budget_curves")


def fig2():
    cur = pd.read_csv(IN / "T_R2A_E3_curve.csv")
    uni = pd.read_csv(D["stats"] / "T_E1_deltaC.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, y in zip(axes, (2025, 2024)):
        for trk, col, mk, off in ((37, C_T37, "o", -0.18), (88, C_T88, "s", 0.18)):
            c = cur[(cur.year == y) & (cur.track == trk)].sort_values("M")
            x = c.N + off
            ax.errorbar(x, 100 * c.dC, yerr=[100 * (c.dC - c.dC_lo), 100 * (c.dC_hi - c.dC)], fmt=mk, ms=3.6,
                        color=col, ecolor=col, elinewidth=0.9, capsize=1.6, label=f"T{trk}, random dates")
            u = uni[(uni.year == y) & (uni.track == trk)].sort_values("M")
            ax.plot(u.N + off, 100 * u.dC, mk, ms=3.8, mfc="white", mec=col, mew=0.8, ls="none",
                    label=f"T{trk}, uniform design")
        ax.axhline(0, color=C_GREY, lw=0.8)
        ax.axhline(1, color=C_GREY, lw=0.6, ls=(0, (2, 2)))
        ax.text(14.0, 1.05, "1 pp", fontsize=6.2, color="#666666", va="bottom", ha="center")
        ax.set_title(f"{y}", fontsize=7.4, pad=3)
        ax.set_xlabel("scenes N (= 2 \u00d7 pairs M)")
        ax.set_xticks([2, 4, 6, 8, 12, 16, 18])
        grid(ax, "y")
    axes[0].set_ylabel(r"coherence gain $\Delta_C$ (pp BA)")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(k) for k in ("T37, random dates", "T37, uniform design", "T88, random dates", "T88, uniform design")]
    h, l = [h[i] for i in order], [l[i] for i in order]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.6)
    letters(fig, axes)
    save(fig, "FigR2_2_coherence_gain")


def fig3():
    ec = pd.read_csv(IN / "T_R2D_equal_cost_E3.csv")
    rows = ec.sort_values(["year", "credits", "track"], ascending=[False, True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.7 + 0.2 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for i, r in rows.iterrows():
        col = C_T37 if r.track == 37 else C_T88
        ax.plot([100 * r.lo, 100 * r.hi], [y[i], y[i]], color=col, lw=1.2, solid_capstyle="round")
        ax.plot([100 * r["diff"]], [y[i]], "o" if r.track == 37 else "s", color=col, ms=3.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(r.year)} T{int(r.track)}  {int(r.credits)} cr  (O3 M={int(r.O3_M)} vs O2 M={int(r.O2_M)})"
                        for r in rows.itertuples()],
                       fontsize=6.4)
    ax.axvline(0, color=C_GREY, lw=0.8)
    ax.set_xlabel("O3 at M pairs  minus  O2 at 1.5 M pairs  (pp BA)")
    grid(ax, "x")
    save(fig, "FigR2_3_equal_cost")


def fig4():
    p = IN / "T_R2E_two_tracks_E5r.csv"
    if not p.exists() or p.stat().st_size < 10:
        print("E5r table empty: FigR2_4 skipped")
        return
    q = pd.read_csv(p)
    if q.empty:
        print("E5r table empty: FigR2_4 skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, y in zip(axes, (2025, 2024)):
        g = q[(q.year == y) & (q.obs == "O2")].sort_values("M_total")
        for trk, col, mk, off in ((37, C_T37, "o", -0.12), (88, C_T88, "s", 0.12)):
            ax.errorbar(g.M_total + off, 100 * g[f"diff_vs_T{trk}"],
                        yerr=[100 * (g[f"diff_vs_T{trk}"] - g[f"lo_T{trk}"]), 100 * (g[f"hi_T{trk}"] - g[f"diff_vs_T{trk}"])],
                        fmt=mk, ms=3.6, color=col, elinewidth=0.9, capsize=1.6, label=f"two tracks minus T{trk} only")
        ax.axhline(0, color=C_GREY, lw=0.8)
        ax.set_title(f"{y}", fontsize=7.4, pad=3)
        ax.set_xlabel("total pairs M")
        ax.set_xticks([2, 4, 6, 8])
        grid(ax, "y")
    axes[0].set_ylabel("BA difference, O2 (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.6)
    letters(fig, axes)
    save(fig, "FigR2_4_two_tracks")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
