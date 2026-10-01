"""Step 8: draft figures for the A1 manuscript from RUN/stats tables (PNG 300 dpi + PDF).

Figures are drafts for internal checking; final styling is done at writing time.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from a1_common import D, TARGET_CLASSES, TRACKS, ensure_dirs, update_status  # noqa: E402

COL = {"O1": "#9aa5b1", "O2": "#2f6db3", "O3": "#d1603d", "O5": "#6a9f58"}
LAB = {"O1": "VV intensity", "O2": "VV+VH intensity", "O3": "VV+VH + 12-day coherence", "O5": "coherence only"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(D["fig"] / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_budget(curve):
    for y in sorted(curve.year.unique(), reverse=True):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, trk in zip(axs, TRACKS):
            c = curve[(curve.year == y) & (curve.track == trk)]
            for obs in ("O1", "O2", "O3", "O5"):
                g = c[c.obs == obs].sort_values("M")
                if len(g):
                    ax.plot(g.N, g.BA * 100, "-o", ms=3, color=COL[obs], label=LAB[obs])
                    ax.fill_between(g.N, g.lo * 100, g.hi * 100, color=COL[obs], alpha=0.15, lw=0)
            ax.set_title(f"T{trk} ({y})")
            ax.set_xlabel("scenes N (= 2 x pairs M)")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("balanced accuracy (%)")
        axs[1].legend(fontsize=7, loc="lower right")
        save(fig, f"Fig3_budget_curve_{y}")


def fig_delta(dcs):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for (y, trk), g in dcs.groupby(["year", "track"]):
        g = g.sort_values("M")
        ls = "-" if y == 2025 else "--"
        ax.errorbar(g.N + (0.15 if trk == 88 else -0.15), g.dC * 100, yerr=[(g.dC - g.lo) * 100, (g.hi - g.dC) * 100],
                    fmt="o" + ls, ms=3, capsize=2, label=f"T{trk} {y}")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("scenes N")
    ax.set_ylabel("coherence gain dC (pp BA)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, "Fig4_coherence_gain")


def fig_perclass(pc):
    for y in sorted(pc.year.unique(), reverse=True):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for ax, trk in zip(axs, TRACKS):
            g = pc[(pc.year == y) & (pc.track == trk)].pivot(index="cls", columns="M", values="dF1").reindex(TARGET_CLASSES)
            im = ax.imshow(g.values * 100, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
            ax.set_xticks(range(g.shape[1]), [2 * m for m in g.columns])
            ax.set_yticks(range(len(g.index)), g.index)
            ax.set_xlabel("scenes N")
            ax.set_title(f"T{trk} ({y})")
        fig.colorbar(im, ax=axs, label="dF1 (pp)")
        save(fig, f"Fig5_perclass_gain_{y}")


def fig_random(e3, curve):
    if not len(e3):
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, trk in zip(axs, TRACKS):
        g = e3[(e3.year == 2025) & (e3.track == trk)].sort_values("M")
        ax.fill_between(2 * g.M, g.O2_p5 * 100, g.O2_p95 * 100, color=COL["O2"], alpha=0.2, label="O2 random draws (5-95%)")
        ax.plot(2 * g.M, g.O2_uniform * 100, "o-", color=COL["O2"], ms=3, label="O2 uniform")
        ax.plot(2 * g.M, g.O3_mean * 100, "s--", color=COL["O3"], ms=3, label="O3 mean of draws")
        ax.set_title(f"T{trk} (2025)")
        ax.set_xlabel("scenes N")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("balanced accuracy (%)")
    axs[1].legend(fontsize=7)
    save(fig, "Fig6_date_selection")


def fig_cost(curve):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for trk, mk in zip(TRACKS, ("o", "s")):
        c = curve[(curve.year == 2025) & (curve.track == trk)]
        for obs, cost in (("O2", 30), ("O3", 45)):
            g = c[c.obs == obs].sort_values("M")
            ax.plot(g.M * cost, g.BA * 100, "-" + mk, ms=3, color=COL[obs], label=f"{LAB[obs]} T{trk}")
    ax.set_xlabel("HyP3 credits (15 per RTC scene, 15 per pair)")
    ax.set_ylabel("balanced accuracy (%)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    save(fig, "Fig9_cost")


def main() -> None:
    ensure_dirs()
    s = D["stats"]
    curve = pd.read_csv(s / "T_E1_curve.csv")
    dcs = pd.read_csv(s / "T_E1_deltaC.csv")
    fig_budget(curve)
    fig_delta(dcs)
    pc = pd.read_csv(s / "T_E1_perclass.csv")
    if len(pc):
        fig_perclass(pc)
    p3 = s / "T_E3_random_draws.csv"
    if p3.exists() and p3.stat().st_size > 5:
        try:
            fig_random(pd.read_csv(p3), curve)
        except pd.errors.EmptyDataError:
            pass
    fig_cost(curve)
    update_status("s8_figures", "DONE")


if __name__ == "__main__":
    main()
