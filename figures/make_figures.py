"""Manuscript figures, generated only from result files (paths below). Output: ../manuscript/figures/*.pdf

Style: IEEE column width 3.5 in / page width 7.16 in, 8 pt text, thin marks, recessive grid, legends for >= 2
series, categorical slots in fixed order from the validated reference palette (blue, orange, aqua).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SAR = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
A0 = SAR / "A0_gamma0_rerun_20260929" / "ws"
S2 = SAR / "site2_20260930" / "runs"
A1 = SAR / "A1_observation_budget_20260923" / "runs" / "20260924_A1_MAIN"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL, PAGE = 3.5, 7.16

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans",
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_classes():
    """Per-class cross-track recall, I28 vs I28+g12, RF, both sites."""
    a = pd.read_csv(A0 / "tables" / "T1_per_class.csv")
    a = a[a.learner == "RF"]
    b = pd.read_csv(S2 / "STUDY1" / "ws" / "tables" / "R1_per_class.csv")
    b = b[b.learner == "RF"]
    classes = ["Forest", "Grassland", "Cropland", "Built-up", "Water"]
    site = {"Site A": [(a.loc[a.class_name == ("Urban" if c == "Built-up" else c), "recall_I28"].iloc[0],
                        a.loc[a.class_name == ("Urban" if c == "Built-up" else c), "recall_I28_G29"].iloc[0]) for c in classes],
            "Site B": [(b[(b["class"] == c) & (b.representation == "I28")].recall.iloc[0],
                        b[(b["class"] == c) & (b.representation == "I28_G29")].recall.iloc[0]) for c in classes]}
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    for ax, (name, vals) in zip(axs, site.items()):
        yy = np.arange(len(classes))[::-1]
        for y, (v0, v1) in zip(yy, vals):
            ax.plot([v0, v1], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([v[0] for v in vals], yy, s=22, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8, label="$I_{28}$")
        ax.scatter([v[1] for v in vals], yy, s=22, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.8, label="$I_{28}{+}g_{12}$")
        ax.set_yticks(yy, classes)
        ax.set_xlim(0, 100)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("Cross-track recall (%)")
        grid(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.08), handletextpad=0.2)
    save(fig, "fig_s1_classes")


def fig_capacity():
    """(a) Coherence increment by learner and site; (b) Site A CNN patch-size trend."""
    t1a = pd.read_csv(A0 / "tables" / "T1_contrasts.csv")
    t1b = pd.read_csv(S2 / "STUDY1" / "ws" / "tables" / "R1_contrasts.csv")
    cnn = pd.read_csv(A0 / "tables_cnn" / "contrasts_checks.csv")
    cnn = cnn[(cnn.contrast == "CNN_IG - CNN_I") & (cnn["eval"] == "sample")]
    # Site-B CNN (tuned) from RESULTS_SITE2.md values recomputed by s2_15 (stored for traceability)
    cnn_b = json.loads((Path(__file__).parent / "site2_cnn_contrast.json").read_text())
    rows = []
    for site, t in (("Site A", t1a), ("Site B", t1b)):
        for l in ("RF", "SVC"):
            if site == "Site A":
                r = t[(t.contrast_id == "C-main") & (t.learner == l) & (t.metric == "cross_ba")].iloc[0]
                rows.append((site, l, r.point_estimate, r.ci_lo, r.ci_hi))
            else:
                r = t[(t.contrast == "C-main") & (t.learner == l) & (t.metric == "cross_ba")].iloc[0]
                rows.append((site, l, r.delta, r.ci_lo, r.ci_hi))
    r = cnn[cnn.family == "tuned_p9"].iloc[0]
    rows.append(("Site A", "CNN", r.delta_cross_ba, r.ci_lo, r.ci_hi))
    rows.append(("Site B", "CNN", cnn_b["delta"], cnn_b["ci_lo"], cnn_b["ci_hi"]))
    fig, axs = plt.subplots(1, 2, figsize=(PAGE * 0.62, 1.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    learners = ["RF", "SVC", "CNN"]
    for k, (site, color, off) in enumerate((("Site A", BLUE, -0.12), ("Site B", ORANGE, 0.12))):
        sub = [r for r in rows if r[0] == site]
        x = [learners.index(s[1]) + off for s in sub]
        y = [s[2] for s in sub]
        e = [[s[2] - s[3] for s in sub], [s[4] - s[2] for s in sub]]
        ax.errorbar(x, y, yerr=e, fmt="o", ms=4.5, color=color, ecolor=color, elinewidth=1, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.6, label=site)
    ax.set_xticks(range(3), ["RF", "SVC", "CNN $9{\\times}9$"])
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("Coherence increment (pp BA)")
    ax.set_ylim(-1.5, 9.5)
    ax.set_title("(a) Learner", loc="left")
    ax.legend(frameon=False, loc="upper right")
    grid(ax, "y")
    ax = axs[1]
    fx = cnn[cnn.family.isin(["fixed_p5", "fixed_p9", "fixed_p11"])]
    p = [5, 9, 11]
    ax.errorbar(p, fx.delta_cross_ba, yerr=[fx.delta_cross_ba - fx.ci_lo, fx.ci_hi - fx.delta_cross_ba], fmt="o-",
                ms=4.5, color=BLUE, elinewidth=1, linewidth=1.2, capsize=0, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(p, ["5×5", "9×9", "11×11"])
    ax.set_xlabel("CNN patch (Site A, fixed training)")
    ax.set_ylim(-1.5, 9.5)
    ax.set_title("(b) Spatial context", loc="left")
    grid(ax, "y")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_capacity")


def fig_bag():
    """Change in built-up share on BAG building pixels by cluster size; RF and CNN; both sites."""
    sa = pd.read_csv(A0 / "bag_check" / "sensitivity_results.csv")
    sb = pd.read_csv(S2 / "STUDY1" / "ws" / "bag_check" / "bag_results.csv")
    sets = [("iso", "1–4"), ("small", "5–50"), ("settle", ">50")]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
    for ax, (name, s) in zip(axs, (("Site A", sa), ("Site B", sb))):
        for fam, color, off in (("RF", BLUE, -0.12), ("CNN", ORANGE, 0.12)):
            v = [s[(s.family == fam) & (s["set"] == k)].iloc[0] for k, _ in sets]
            x = np.arange(3) + off
            ax.errorbar(x, [r.delta_pp for r in v], yerr=[[r.delta_pp - r.ci_lo for r in v], [r.ci_hi - r.delta_pp for r in v]],
                        fmt="o", ms=4.5, color=color, elinewidth=1, capsize=0, markeredgecolor="white", markeredgewidth=0.6,
                        label=fam)
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_xticks(range(3), [lab for _, lab in sets])
        ax.set_xlabel("Building cluster (cells)")
        ax.set_title(name, loc="left")
        grid(ax, "y")
    axs[0].set_ylabel("Change in built-up share (pp)")
    axs[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_bag")


def fig_budget():
    """Delta_C vs number of pairs (E3), both sites."""
    sites = (("Site A", A1 / "stats_r2" / "T_R2A_E3_curve.csv"), ("Site B", S2 / "MAIN" / "stats_r2" / "T_R2A_E3_curve.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(PAGE * 0.62, 2.0), sharey=True)
    for ax, (name, p) in zip(axs, sites):
        e = pd.read_csv(p)
        for (trk, color), (yr, ls, off) in [((t, c), (y, l, o)) for t, c in ((37, BLUE), (88, ORANGE))
                                              for y, l, o in ((2025, "-", -0.08), (2024, "--", 0.08))]:
            g = e[(e.track == trk) & (e.year == yr)].sort_values("M")
            x = g.M + off + (0.04 if trk == 88 else -0.04)
            ax.errorbar(x, g.dC * 100, yerr=[(g.dC - g.dC_lo) * 100, (g.dC_hi - g.dC) * 100], fmt="o", linestyle=ls,
                        ms=3.2, color=color, elinewidth=0.8, linewidth=1.1, capsize=0, markeredgecolor="white",
                        markeredgewidth=0.5, label=f"T{trk} {yr}")
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_xticks([1, 2, 3, 4, 6, 8, 9])
        ax.set_xlabel("Repeat pairs $M$ (scenes $2M$)")
        ax.set_title(name, loc="left")
        grid(ax, "y")
    axs[0].set_ylabel("Coherence gain $\\Delta_C$ (pp BA)")
    fig.tight_layout(w_pad=0.8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), columnspacing=1.2, handlelength=2.2)
    save(fig, "fig_budget")


def fig_cost_pol():
    """(a) Equal-cost contrast vs credits; (b) polarization gain vs coherence gain (uniform design)."""
    fig, axs = plt.subplots(1, 2, figsize=(PAGE * 0.62, 2.0))
    ax = axs[0]
    for name, p, color, off in (("Site A", A1 / "stats_r2" / "T_R2D_equal_cost_E3.csv", BLUE, -6),
                                ("Site B", S2 / "MAIN" / "stats_r2" / "T_R2D_equal_cost_E3.csv", ORANGE, 6)):
        c = pd.read_csv(p).sort_values(["credits", "year", "track"])
        jitter = np.tile([-4.5, -1.5, 1.5, 4.5], len(c) // 4)  # 4 year-track cells per credit level, side by side
        ax.errorbar(c.credits + 2 * off + jitter, c["diff"] * 100, yerr=[(c["diff"] - c.lo) * 100, (c.hi - c["diff"]) * 100],
                    fmt="o", ms=3.2, color=color, elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.5,
                    label=name)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks([90, 180, 270])
    ax.set_xlabel("HyP3 credits")
    ax.set_ylabel("$O_3$ at $M$ minus $O_2$ at $1.5M$ (pp BA)")
    ax.set_title("(a) Equal cost", loc="left")
    ax.legend(frameon=False, loc="lower right")
    grid(ax, "y")
    ax = axs[1]
    for name, p, color in (("Site A", A1 / "stats" / "T_E1_deltaC.csv", BLUE), ("Site B", S2 / "MAIN" / "stats" / "T_E1_deltaC.csv", ORANGE)):
        d = pd.read_csv(p)
        ax.scatter(d.dC * 100, d.dPol * 100, s=14, color=color, edgecolor="white", linewidth=0.5, label=name, zorder=3)
    lim = [-1, 19]
    ax.plot(lim, lim, color=INK2, linewidth=0.6, linestyle=":")
    ax.text(12.5, 11.0, "equal gain", color=INK2, fontsize=6.5, rotation=38)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 19)
    ax.set_xlabel("Coherence gain $\\Delta_C$ (pp)")
    ax.set_ylabel("Polarization gain $\\Delta_P$ (pp)")
    ax.set_title("(b) Observable", loc="left")
    grid(ax, "both")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_cost_pol")


def fig_sites():
    """(a) Site locations; (b, c) 12-day VV coherence, T15 spring 2024, at Site A and Site B."""
    import geopandas as gpd
    from shapely.geometry import box
    ne = gpd.read_file(f"zip://{Path(__file__).parent / 'ne' / 'ne_10m_admin_0_countries.zip'}")
    win = box(2.5, 50.6, 8.0, 53.8)
    land = ne[ne.intersects(win)].clip(win)
    sites = {"A": (6.00, 51.98, 6.54, 52.38), "B": (6.55, 52.75, 7.05, 53.10)}
    ga = np.load(SAR / "A0_gamma0_rerun_20260929" / "data" / "cube_4track_v7.npz")["t15_g12"]
    gb = np.load(S2 / "STUDY1" / "cube_spring.npz")["t15_g12"]
    fig = plt.figure(figsize=(PAGE, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    land[land.ADMIN != "Netherlands"].plot(ax=ax, color="#f1f0ec", edgecolor="#c9c8c2", linewidth=0.4)
    land[land.ADMIN == "Netherlands"].plot(ax=ax, color="#e4e3df", edgecolor=INK2, linewidth=0.5)
    for k, (x0, y0, x1, y1) in sites.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ORANGE, linewidth=1.4))
        ax.text(x0 - 0.08, y1 + 0.05, f"Site {k}", color=INK, fontsize=7, ha="right", va="bottom")
    ax.set_xlim(3.2, 7.4)
    ax.set_ylim(50.7, 53.65)
    ax.set_aspect(1 / np.cos(np.radians(52.2)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Sites", loc="left")
    ax.grid(color=GRID, linewidth=0.4)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("coh", ["#f4f7fb", "#9ec3ec", BLUE, "#0d3a73"])
    for i, (name, g) in enumerate((("(b) Site A", ga), ("(c) Site B", gb))):
        ax = fig.add_subplot(gs[i + 1])
        g = np.where(g > 0, g, np.nan)
        h, w = g.shape
        im = ax.imshow(g, cmap=cmap, vmin=0, vmax=1, extent=[0, w * 0.04, 0, h * 0.04], interpolation="nearest")
        ax.set_title(name, loc="left")
        ax.set_xlabel("Easting (km)")
        if i == 0:
            ax.set_ylabel("Northing (km)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.02)
    cb.set_label("$g_{12}$")
    cb.outline.set_linewidth(0.5)
    save(fig, "fig_sites")


if __name__ == "__main__":
    fig_sites()
    fig_classes()
    fig_capacity()
    fig_bag()
    fig_budget()
    fig_cost_pol()
