"""Reproducible visual additions for the merged A0+A1 manuscript.

The protocol panel is a diagram. Image and distribution panels use the
frozen Site A/Site B spring cubes; no model is trained by this script.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from figure_typography import normalize_figure
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SITE_A = SAR / "A0_gamma0_rerun_20260929" / "data" / "cube_4track_v7.npz"
SITE_B = SAR / "site2_20260930" / "runs" / "STUDY1" / "cube_spring.npz"
SITE_B_LABELS = SAR / "site2_20260930" / "runs" / "STUDY1" / "labels.npz"
SITE_A_MMU = SAR / "SAR_v2" / "mmu" / "mmu_mask.npz"
SITE_A_CLC = SAR / "SAR_v2" / "labels" / "consensus_mask.npz"
SITE_A_OMEGA = SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy"

BLUE = "#2A78D6"
ORANGE = "#D86732"
PURPLE = "#803A94"
INK = "#192635"
GRID = "#E0E5EA"
CLASSES = ["Forest", "Grassland", "Cropland", "Built-up", "Water"]
CLASS_COLORS = ["#287B57", "#A6BA46", "#E4A13D", "#9A4B8C", "#3B75AE"]

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, name: str) -> None:
    normalize_figure(fig)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{name}.{suffix}", **kwargs)
    plt.close(fig)


def rounded_box(ax, xy, wh, face, edge, title, lines, title_color=INK):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.007,rounding_size=0.018",
            transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.9,
        )
    )
    ax.text(x + 0.020, y + h - 0.045, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, fontweight="bold", color=title_color)
    for j, line in enumerate(lines):
        ax.text(x + 0.020, y + h - 0.115 - j * 0.071, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.6, color=INK)


def arrow(ax, x, y0, y1, color):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1,
                                 color=color, transform=ax.transAxes))


def fig_design() -> None:
    """One message: both studies estimate a paired increment on matched support."""
    fig, ax = plt.subplots(figsize=(7.16, 2.55))
    ax.set_axis_off()
    rounded_box(ax, (0.08, 0.79), (0.84, 0.16), "#F0F3F6", "#9DAAB5",
                "Paired coherence increment", ["Matched acquisitions, reference units, and spatial folds for intensity vs. intensity + coherence"])
    arrow(ax, 0.265, 0.78, 0.69, BLUE)
    arrow(ax, 0.735, 0.78, 0.69, ORANGE)
    rounded_box(ax, (0.035, 0.36), (0.46, 0.31), "#ECF3FB", BLUE,
                "(a) Study 1  |  two acquisitions", [
                    "Four tracks · two sites · five land-cover classes",
                    "Intensity features vs. intensity + coherence",
                    "Pixel RF/SVC; spatial-context CNN; BAG check",
                ])
    rounded_box(ax, (0.505, 0.36), (0.46, 0.31), "#FFF2EB", ORANGE,
                "(b) Study 2  |  seasonal budget", [
                    "One to nine pairs · two tracks · two years",
                    "Seven crop classes; date-design repeats",
                    "HGB/TempCNN; equal processing credits",
                ])
    arrow(ax, 0.265, 0.35, 0.26, BLUE)
    arrow(ax, 0.735, 0.35, 0.26, ORANGE)
    rounded_box(ax, (0.035, 0.06), (0.46, 0.18), "#FFFFFF", "#A5B5C5",
                "Measured increment", ["Class definition × model context"])
    rounded_box(ax, (0.505, 0.06), (0.46, 0.18), "#FFFFFF", "#C9AF9F",
                "Measured increment", ["Acquisition budget × processing cost"])
    save(fig, "fig_design")


def load_sites():
    with np.load(SITE_A, allow_pickle=False) as z, \
         np.load(SITE_A_MMU, allow_pickle=False) as mmu, \
         np.load(SITE_A_CLC, allow_pickle=False) as clc:
        y = z["y"].copy()
        mv = mmu["mmu_valid"]
        clc_full = np.full(y.shape, -1, dtype=np.int16)
        clc_full[mv] = clc["clc_on_mmu"].astype(np.int16) - 1
        omega = np.load(SITE_A_OMEGA, allow_pickle=False)
        c2 = omega & mv & (y >= 0) & (y < 5) & (y == clc_full)
        if int(c2.sum()) != 287013:
            raise ValueError(f"Site A C2 support changed: {int(c2.sum())}")
        a = {
            "vv": z["t15_d1_vv"].copy(),
            "g": z["t15_g12"].copy(),
            "y": y,
            "image_mask": z["t15_valid"].copy(),
            "label_mask": c2,
        }
    with np.load(SITE_B, allow_pickle=False) as z, np.load(SITE_B_LABELS, allow_pickle=False) as labels:
        y = labels["y"].copy()
        clc = labels["clc"].astype(np.int16) - 1
        c2 = labels["omega"] & (y >= 0) & (y < 5) & (y == clc)
        if int(c2.sum()) != 505806:
            raise ValueError(f"Site B C2 support changed: {int(c2.sum())}")
        b = {
            "vv": z["t15_d1_vv"].copy(),
            "g": z["t15_g12"].copy(),
            "y": y,
            "image_mask": z["t15_valid"].copy(),
            "label_mask": c2,
        }
    return {"Site A": a, "Site B": b}


def crop_for_site(site: dict, pixels: int = 160):
    # A fixed, reproducible illustration rule: center on the spatial median
    # of the mapped built-up support. It never uses model predictions.
    coords = np.argwhere(site["label_mask"] & (site["y"] == 3))
    row, col = np.median(coords, axis=0).astype(int)
    h, w = site["y"].shape
    r0 = int(np.clip(row - pixels // 2, 0, h - pixels))
    c0 = int(np.clip(col - pixels // 2, 0, w - pixels))
    return (r0, r0 + pixels, c0, c0 + pixels)


def fig_detail(sites: dict) -> dict:
    """Real VV, coherence and harmonized class support on matched 6.4-km ROIs."""
    cmap_class = ListedColormap(CLASS_COLORS)
    cmap_class.set_bad("#FFFFFF")
    norm_class = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap_class.N)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.65))
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.145, top=0.94,
                        wspace=0.07, hspace=0.18)
    detail = {}
    im_vv = im_g = None
    for row, (name, site) in enumerate(sites.items()):
        r0, r1, c0, c1 = crop_for_site(site)
        detail[name] = {"crop_rows": [r0, r1], "crop_cols": [c0, c1],
                        "roi_width_km": round((c1 - c0) * 0.04, 2)}
        sl = np.s_[r0:r1, c0:c1]
        image_valid = site["image_mask"][sl]
        label_valid = site["label_mask"][sl]
        vv = np.ma.masked_where(~image_valid | ~np.isfinite(site["vv"][sl]), site["vv"][sl])
        g = np.ma.masked_where(~image_valid | ~np.isfinite(site["g"][sl]), site["g"][sl])
        cls = np.ma.masked_where(~label_valid, site["y"][sl])
        im_vv = axes[row, 0].imshow(vv, cmap="Greys_r", vmin=-18, vmax=-2,
                                     interpolation="nearest")
        im_g = axes[row, 1].imshow(g, cmap="cividis", vmin=0, vmax=1,
                                  interpolation="nearest")
        axes[row, 2].imshow(cls, cmap=cmap_class, norm=norm_class,
                            interpolation="nearest")
        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({chr(97 + row * 3 + col)}) {name}: " +
                         ("VV backscatter" if col == 0 else "VV coherence" if col == 1 else "Reference classes"),
                         loc="left", fontweight="bold", pad=4)
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.4)
        # 25 pixels at 40 m spacing = 1 km.
        ax = axes[row, 2]
        ax.plot([122, 147], [144, 144], color="black", lw=2.2, solid_capstyle="butt")
        ax.text(134.5, 139, "1 km", color="black", ha="center", va="bottom",
                fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
    cax_vv = fig.add_axes([0.075, 0.084, 0.21, 0.013])
    cax_g = fig.add_axes([0.395, 0.084, 0.21, 0.013])
    fig.colorbar(im_vv, cax=cax_vv, orientation="horizontal", ticks=[-18, -10, -2]).set_label("$\\gamma^0_{VV}$ (dB)", fontsize=7)
    fig.colorbar(im_g, cax=cax_g, orientation="horizontal", ticks=[0, 0.5, 1]).set_label("$g_{12}$", fontsize=7)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=n) for c, n in zip(CLASS_COLORS, CLASSES)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.81, 0.055),
               ncol=3, frameon=False, fontsize=6.5, columnspacing=0.6, handlelength=1.0)
    save(fig, "fig_detail")
    return detail


def fig_signature(sites: dict) -> dict:
    """Descriptive class signatures; displayed KDE uses a fixed capped sample."""
    rng = np.random.default_rng(20261002)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.88, wspace=0.14)
    stats = {}
    for ax, (name, site) in zip(axes, sites.items()):
        valid = site["label_mask"] & np.isfinite(site["g"])
        arrs = []
        stats[name] = {}
        for code, label in enumerate(CLASSES):
            x = site["g"][valid & (site["y"] == code)]
            ix = rng.choice(len(x), size=min(len(x), 6000), replace=False)
            arrs.append(x[ix])
            stats[name][label] = {"n_all": int(len(x)), "n_plotted": int(len(ix)),
                                  "median_all": round(float(np.median(x)), 4)}
        parts = ax.violinplot(arrs, positions=np.arange(5), widths=0.78,
                              showextrema=False, showmedians=False, points=70)
        for body, color in zip(parts["bodies"], CLASS_COLORS):
            body.set_facecolor(color)
            body.set_edgecolor(INK)
            body.set_linewidth(0.4)
            body.set_alpha(0.82)
        medians = [stats[name][label]["median_all"] for label in CLASSES]
        ax.scatter(np.arange(5), medians, s=20, marker="o", facecolor="white",
                   edgecolor=INK, linewidth=0.9, zorder=3)
        ax.set_xticks(np.arange(5), CLASSES, rotation=22, ha="right")
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"({'a' if name == 'Site A' else 'b'}) {name}", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("12-day VV coherence $g_{12}$")
    fig.text(0.52, 0.035, "Open circles: median of all valid reference pixels", ha="center",
             fontsize=7, color=INK)
    save(fig, "fig_signature")
    return stats


def main():
    fig_design()
    sites = load_sites()
    rois = fig_detail(sites)
    stats = fig_signature(sites)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256_before_manifest": sha256(Path(__file__)),
        "inputs": {str(p): sha256(p) for p in (SITE_A, SITE_B, SITE_B_LABELS, SITE_A_MMU, SITE_A_CLC, SITE_A_OMEGA)},
        "roi_rule": "160x160 pixels, centered at spatial median of C2-consensus built-up pixels; clipped to grid",
        "label_support": {"Site A C2": 287013, "Site B C2": 505806},
        "rois": rois,
        "signature_sampling": "up to 6000 pixels per site/class, uniform without replacement, seed 20261002; medians use all valid pixels",
        "class_signature": stats,
        "outputs": [str(OUT / f"{name}.{suffix}") for name in ("fig_design", "fig_detail", "fig_signature")
                    for suffix in ("pdf", "svg", "png")],
    }
    (HERE / "visual_additions_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
