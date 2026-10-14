"""Fig. 2: processing chain and the two paired experiments (plain IEEE-style line diagram).

Thumbnails are real Site A T15 layers on the 6.4 km window of Fig. 3 (same crop rule as
make_visual_additions.crop_for_site). The timeline shows the nine disjoint 12-day pairs of
track T37 in 2025 (site2_20260930/manifests/budget_design.csv, M = 9 row; same dates at both sites).
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

from figure_typography import normalize_figure
from make_visual_additions import CLASS_COLORS, crop_for_site, load_sites

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
BUDGET = SAR / "site2_20260930" / "manifests" / "budget_design.csv"
INK, INK2, ACC = "#1b1b1b", "#5a5a5a", "#b03a2e"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "mathtext.fontset": "stix"})


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def box(ax, x, y, w, h, lines, edge=INK, lw=0.7, head=None, fs=7.4):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=edge, linewidth=lw, zorder=2))
    n = len(lines) + (1 if head else 0)
    step = 0.060
    y0 = y + h / 2 + (n - 1) * step / 2
    k = 0
    if head:
        ax.text(x + w / 2, y0, head, ha="center", va="center", fontsize=fs + 0.4, fontstyle="italic", zorder=3)
        k = 1
    for j, s in enumerate(lines):
        ax.text(x + w / 2, y0 - (j + k) * step, s, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p0, p1, color=INK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def main():
    sites = load_sites()
    a = sites["Site A"]
    r0, r1, c0, c1 = crop_for_site(a)
    sl = np.s_[r0:r1, c0:c1]
    vv = np.ma.masked_where(~a["image_mask"][sl], a["vv"][sl])
    g = np.ma.masked_where(~a["image_mask"][sl], a["g"][sl])
    cls = np.ma.masked_where(~a["label_mask"][sl], a["y"][sl])

    rows = [r for r in csv.DictReader(BUDGET.open()) if r["year"] == "2025" and r["track"] == "37" and r["M"] == "9"]
    assert len(rows) == 1
    pairs = sorted(p.split("_")[5:7] for p in rows[0]["pair_ids"].split(";"))
    assert len(pairs) == 9

    W, H = 7.16, 3.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # shared processing column
    box(ax, 0.008, 0.555, 0.155, 0.36, ["Sentinel-1 IW SLC", "4 tracks: T15, T88 (asc.)", "T37, T139 (desc.)",
                                        "2 sites, 2024–2025"])
    box(ax, 0.008, 0.085, 0.155, 0.36, ["ASF HyP3 (GAMMA)", "RTC $\\gamma^0$, 20 m", "InSAR $g_{12}$, 10×2 looks",
                                        "40 m grid, linear-power", "area averaging"])
    arrow(ax, (0.0855, 0.555), (0.0855, 0.445))

    # lane labels
    ax.text(0.185, 0.965, "Study 1 — two acquisitions, five land-cover classes", ha="left", va="center",
            fontsize=8.3, fontweight="bold")
    ax.text(0.185, 0.475, "Study 2 — seasonal budget, seven crop classes", ha="left", va="center",
            fontsize=8.3, fontweight="bold")
    ax.plot([0.18, 0.995], [0.50, 0.50], color="#c8c8c8", lw=0.5)
    ax.plot([0.163, 0.174, 0.174], [0.265, 0.265, 0.735], color=INK, lw=0.8)
    arrow(ax, (0.174, 0.735), (0.186, 0.735))
    arrow(ax, (0.174, 0.265), (0.192, 0.265))

    # Study 1 thumbnails
    cm = ListedColormap(CLASS_COLORS)
    cm.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    th = [(vv, "Greys_r", None, -18, -2, "$\\gamma^0_{VV}$"), (g, "cividis", None, 0, 1, "$g_{12}$"),
          (cls, cm, norm, None, None, "Reference")]
    tw, thh = 0.083, 0.083 * W / H
    for j, (arr, cmap, nm, vmin, vmax, lab) in enumerate(th):
        x = 0.188 + j * (tw + 0.008)
        tax = fig.add_axes([x, 0.585, tw, thh])
        tax.imshow(arr, cmap=cmap, norm=nm, vmin=vmin, vmax=vmax, interpolation="nearest")
        tax.set_xticks([])
        tax.set_yticks([])
        for s in tax.spines.values():
            s.set_linewidth(0.5)
        ax.text(x + tw / 2, 0.572, lab, ha="center", va="top", fontsize=7.2)
    xs = 0.188 + 3 * tw + 2 * 0.008
    arrow(ax, (xs + 0.006, 0.735), (0.478, 0.735))
    box(ax, 0.478, 0.585, 0.17, 0.30, ["$I_{28}$  vs.  $I_{28}{+}g_{12}$", "RF, SVC (pixel)",
                                       "$I$  vs.  $I{+}g_{12}$, CNN", "(1×1 to 11×11 patches)"],
        edge=ACC, lw=1.1, head="paired inputs")
    arrow(ax, (0.648, 0.735), (0.668, 0.735))
    box(ax, 0.668, 0.585, 0.16, 0.30, ["5 × 5 km blocks", "400 m exclusion", "train on 1 track,",
                                       "test on 3 others"], head="spatial transfer")
    arrow(ax, (0.828, 0.735), (0.848, 0.735))
    box(ax, 0.848, 0.585, 0.145, 0.30, ["$\\Delta$BA, class recall", "block bootstrap", "BAG building",
                                        "register check"], head="output")

    # Study 2 timeline: nine disjoint 12-day pairs, T37 2025
    tl = fig.add_axes([0.195, 0.165, 0.265, 0.20])
    d0 = date(2025, 3, 1).toordinal()
    for k, (s, e) in enumerate(pairs):
        ds = date(int(s[:4]), int(s[4:6]), int(s[6:])).toordinal() - d0
        de = date(int(e[:4]), int(e[4:6]), int(e[6:])).toordinal() - d0
        tl.plot([ds, de], [0, 0], color=INK, lw=2.2, solid_capstyle="butt")
        tl.plot([ds, de], [0, 0], "o", ms=2.6, color=INK)
    tl.axhline(0, color="#bdbdbd", lw=0.6, zorder=0)
    months = [(m, date(2025, m, 1).toordinal() - d0) for m in range(3, 12)]
    tl.set_xticks([v for _, v in months], ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])
    tl.set_yticks([])
    tl.set_ylim(-1, 1)
    tl.set_xlim(-3, months[-1][1])
    tl.tick_params(axis="x", labelsize=6.6, length=2, pad=1)
    for s in ("left", "right", "top"):
        tl.spines[s].set_visible(False)
    tl.spines["bottom"].set_linewidth(0.5)
    ax.text(0.3275, 0.405, "9 disjoint 12-day pairs per track and year", ha="center", fontsize=7.2)
    ax.text(0.3275, 0.065, "T37, 2025; $M \\in \\{1,\\ldots,9\\}$ pairs drawn per design", ha="center",
            fontsize=6.6, color=INK2)
    arrow(ax, (0.463, 0.255), (0.478, 0.255))
    box(ax, 0.478, 0.105, 0.17, 0.30, ["$O_2$: VV, VH, VH$-$VV", "at 2$M$ scenes", "$O_3$: $O_2$ + $M$ coherences",
                                       "HGB, RF, LR, TempCNN"], edge=ACC, lw=1.1, head="paired inputs")
    arrow(ax, (0.648, 0.255), (0.668, 0.255))
    box(ax, 0.668, 0.105, 0.16, 0.30, ["BRP parcels", "random date designs", "6 spatial folds"],
        head="parcel evaluation")
    arrow(ax, (0.828, 0.255), (0.848, 0.255))
    box(ax, 0.848, 0.105, 0.145, 0.30, ["$\\Delta_C(M)$ vs. budget", "equal cost:", "$O_3(M)$ vs. $O_2(1.5M)$"],
        head="output")

    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_workflow.{ext}", **kw)
    plt.close(fig)
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
           "inputs_sha256": {str(BUDGET): sha(BUDGET)}, "crop_rows_cols": [r0, r1, c0, c1],
           "timeline_pairs_T37_2025": pairs}
    (HERE / "fig_workflow_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(pairs)


if __name__ == "__main__":
    main()
