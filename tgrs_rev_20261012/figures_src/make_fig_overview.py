"""Fig. 1: study sites, Sentinel-1 track coverage, and Study 1 spatial blocks.

(a) Netherlands with the footprints of the spring-2024 SLC scenes used at both sites (ASF search API,
    queried 2026-10-12; scene names match site2_20260930/manifests/study1_pairs.csv and the Site A
    HyP3 products). (b, c) Full analysis grids at Site A and Site B: false-colour composite of
    T15 first-date VV (R), VH (G) and 12-day VV coherence (B), with the 5 x 5 km training and
    evaluation blocks of Study 1. No model output is used.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from rasterio.windows import from_bounds as win_from_bounds
from shapely.geometry import box

from figure_typography import normalize_figure

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
CUBE_A = SAR / "A0_gamma0_rerun_20260929" / "data" / "cube_4track_v7.npz"
CUBE_B = SAR / "site2_20260930" / "runs" / "STUDY1" / "cube_spring.npz"
META_B = SAR / "site2_20260930" / "runs" / "STUDY1" / "meta.json"
SPLIT_A = SAR / "A0_gamma0_rerun_20260929" / "ws" / "splits" / "split_manifest.json"
SPLIT_B = SAR / "site2_20260930" / "runs" / "STUDY1" / "split_manifest.json"
REF_A = next((SAR / "sar_v2" / "data" / "products").glob("**/*717E*inc_map_ell.tif"))
NE = HERE.parent.parent / "merged_tgrs_20260930" / "figures_src" / "ne" / "ne_10m_admin_0_countries.zip"
AOI_A = [6.00, 51.98, 6.54, 52.38]
TRACKS = {  # colour, line style, pass
    15: ("#1f5fa8", "-", "Asc."), 88: ("#5aa0e0", "-", "Asc."),
    37: ("#b5471b", (0, (4, 2)), "Desc."), 139: ("#e8954a", (0, (4, 2)), "Desc."),
}
INK, INK2 = "#1b1b1b", "#55544f"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "stix"})


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def grid_transform_a():
    """Identical construction to A0_gamma0_rerun_20260929/scripts/s3_build_cube_v7.py::reference_grid."""
    with rasterio.open(REF_A) as s:
        tf = pyproj.Transformer.from_crs("EPSG:4326", s.crs, always_xy=True)
        x0, y0 = tf.transform(AOI_A[0], AOI_A[1])
        x1, y1 = tf.transform(AOI_A[2], AOI_A[3])
        win = win_from_bounds(x0, y0, x1, y1, s.transform)
        assert (int(round(win.height)), int(round(win.width))) == (1077, 965)
        assert s.crs.to_epsg() == 32632
        return s.window_transform(win)


def grid_transform_b():
    a = json.loads(META_B.read_text())["grid"]["affine_gdal"]
    return rasterio.Affine.from_gdal(*a)


RANGES = {"g12": (0.10, 0.95), "vh": (-27.0, -9.0), "ratio": (2.0, 13.0)}


def composite(z):
    """Coherence-intensity composite of track T15: R = 12-day VV coherence, G = two-date mean VH (dB),
    B = two-date mean VV - VH (dB). Shown wherever T15 is valid; the four-track support is outlined."""
    def st(x, key):
        lo, hi = RANGES[key]
        return np.clip((x - lo) / (hi - lo), 0, 1)
    vv = (z["t15_d1_vv"] + z["t15_d2_vv"]) / 2
    vh = (z["t15_d1_vh"] + z["t15_d2_vh"]) / 2
    rgb = np.dstack([st(z["t15_g12"], "g12"), st(vh, "vh"), st(vv - vh, "ratio")])
    shown = z["t15_valid"].astype(bool) & np.isfinite(rgb).all(axis=2)
    rgba = np.dstack([np.nan_to_num(rgb), shown.astype(float)])
    support = z["four_track_valid"].astype(bool)
    return rgba, support


def draw_site(ax, rgba, support, tf, split, label, scalebar_km=5):
    h, w = rgba.shape[:2]
    x0, y1 = tf * (0, 0)
    x1, y0 = tf * (w, h)
    ax.set_facecolor("#ececea")
    ext = (x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3)
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    if not support.all():
        ax.contour(support.astype(float), levels=[0.5], colors="white", linewidths=0.9, linestyles="--",
                   extent=ext, origin="upper", zorder=4)
    for bid in split["train_blocks"] + split["eval_blocks"]:
        r, c = divmod(bid, 100)
        bx, by = tf * (c * 125, (r + 1) * 125)
        ev = bid in split["eval_blocks"]
        ax.add_patch(Rectangle((bx / 1e3, by / 1e3), 5, 5, fill=False,
                               edgecolor="white" if ev else "#1b1b1b",
                               linewidth=1.5 if ev else 0.35, alpha=1.0 if ev else 0.55, zorder=3 if ev else 2))
    ax.set_xlim(x0 / 1e3, x1 / 1e3)
    ax.set_ylim(y0 / 1e3, y1 / 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 32N)")
    ax.set_title(label, loc="left")
    ax.ticklabel_format(useOffset=False, style="plain")
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    # scale bar and north arrow (UTM grid north) on a white backing box
    sx, sy = x0 / 1e3 + 1.5, y0 / 1e3 + 1.3
    ax.add_patch(Rectangle((sx - 0.8, sy - 0.7), scalebar_km + 1.6, 3.4, facecolor="white", edgecolor="none",
                           alpha=0.85, zorder=5))
    ax.add_patch(Rectangle((sx, sy), scalebar_km, 0.5, facecolor="white", edgecolor="black", lw=0.5, zorder=6))
    ax.add_patch(Rectangle((sx, sy), scalebar_km / 2, 0.5, facecolor="black", edgecolor="black", lw=0.5, zorder=6))
    ax.text(sx + scalebar_km / 2, sy + 0.8, f"{scalebar_km} km", ha="center", va="bottom", fontsize=7, zorder=7)
    nx, ny = x1 / 1e3 - 2.4, y1 / 1e3 - 6.2
    ax.add_patch(Rectangle((nx - 1.4, ny - 0.8), 2.8, 6.0, facecolor="white", edgecolor="none", alpha=0.85,
                           zorder=5))
    ax.add_patch(FancyArrowPatch((nx, ny), (nx, ny + 3.2), arrowstyle="-|>", mutation_scale=8, color="black",
                                 lw=0.9, zorder=6))
    ax.text(nx, ny + 3.3, "N", ha="center", va="bottom", fontsize=7, zorder=7)


def main():
    ne = gpd.read_file(f"zip://{NE}")
    win = box(2.0, 50.0, 9.5, 54.5)
    land = ne[ne.intersects(win)].clip(win)
    fps = {}
    for site in ("A", "B"):
        for t in TRACKS:
            p = HERE / f"asf_site{site}_T{t}.geojson"
            g = gpd.read_file(p)
            g = g[g.processingLevel == "SLC"] if "processingLevel" in g else g
            fps[(site, t)] = (g.iloc[0].geometry, g.iloc[0]["sceneName"], p)

    za, zb = np.load(CUBE_A), np.load(CUBE_B)
    rgba_a, va = composite(za)
    rgba_b, vb = composite(zb)
    tfa, tfb = grid_transform_a(), grid_transform_b()
    sa = json.loads(SPLIT_A.read_text())["historical_development_split"]
    sb = json.loads(SPLIT_B.read_text())["historical_development_split"]
    assert len(sa["train_blocks"]) == 28 and len(sa["eval_blocks"]) == 12
    assert len(sb["train_blocks"]) == 39 and len(sb["eval_blocks"]) == 17

    fig = plt.figure(figsize=(7.16, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.93, 0.93], wspace=0.30,
                          left=0.065, right=0.995, top=0.90, bottom=0.25)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#dfe8ef")
    land[land.ADMIN != "Netherlands"].plot(ax=ax, color="#f3f2ee", edgecolor="#b9b8b2", linewidth=0.4)
    land[land.ADMIN == "Netherlands"].plot(ax=ax, color="#e3e1da", edgecolor=INK2, linewidth=0.55)
    seen = set()
    for (site, t), (geom, name, _) in fps.items():
        if name in seen:
            continue
        seen.add(name)
        col, ls, _ = TRACKS[t]
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.0)
    sites = {"A": AOI_A, "B": [6.55, 52.75, 7.05, 53.10]}
    for k, (x0, y0, x1, y1) in sites.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#c0392b", edgecolor=INK, alpha=0.35,
                               lw=0, zorder=4))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.2, zorder=5))
        ax.text(x0 - 0.07, (y0 + y1) / 2, k, ha="right", va="center", fontsize=8.5, zorder=5)
    ax.set_xlim(3.3, 8.2)
    ax.set_ylim(50.45, 54.0)
    ax.set_aspect(1 / np.cos(np.radians(52.2)))
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_xticklabels([f"{v}°E" for v in (4, 5, 6, 7, 8)])
    ax.set_yticks([51, 52, 53])
    ax.set_yticklabels([f"{v}°N" for v in (51, 52, 53)])
    ax.tick_params(length=2)
    ax.set_title("(a) Sentinel-1 coverage", loc="left")
    # flight (solid) and look (dashed) directions; Sentinel-1 is right-looking. Headings are nominal
    # (about -12 deg ascending, 192 deg descending) and drawn for orientation only.
    k = 1 / np.cos(np.radians(52.2))  # degrees of latitude per degree of longitude on this map
    ax.add_patch(Rectangle((3.36, 50.50), 1.75, 1.05, facecolor="white", edgecolor="#bbbbbb", lw=0.4,
                           alpha=0.9, zorder=6))
    for (x, y, hd, lab) in ((3.80, 51.02, -12.0, "Asc."), (4.66, 51.02, 192.0, "Desc.")):
        th = np.radians(hd)
        fx, fy = 0.26 * np.sin(th), 0.26 * np.cos(th) / k
        ax.add_patch(FancyArrowPatch((x - fx, y - fy), (x + fx, y + fy), arrowstyle="-|>",
                                     mutation_scale=8, color=INK, lw=1.0, zorder=7))
        lx, ly = 0.36 * np.cos(th), -0.36 * np.sin(th) / k
        ax.add_patch(FancyArrowPatch((x, y), (x + lx, y + ly), arrowstyle="-|>", mutation_scale=7,
                                     color="#c0392b", lw=0.8, linestyle=(0, (2, 1.2)), zorder=7))
        ax.text(x, y - 0.31, lab, ha="center", va="top", fontsize=6.5, zorder=7)
    ax.text(4.23, 51.50, "flight / look", ha="center", va="top", fontsize=6, color=INK2, zorder=7)
    handles = [Line2D([], [], color=TRACKS[t][0], linestyle=TRACKS[t][1], lw=1.1,
                      label=f"T{t} ({TRACKS[t][2]})") for t in (15, 88, 37, 139)]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor="#bbbbbb", handlelength=2.2, borderpad=0.35, labelspacing=0.25)

    axa = fig.add_subplot(gs[1])
    draw_site(axa, rgba_a, va, tfa, sa, "(b) Site A")
    axa.set_ylabel("Northing (km, UTM 32N)")
    axb = fig.add_subplot(gs[2])
    draw_site(axb, rgba_b, vb, tfb, sb, "(c) Site B")
    for a_ in (axa, axb):
        a_.locator_params(axis="x", nbins=3)
        a_.locator_params(axis="y", nbins=4)

    leg = [Patch(facecolor="none", edgecolor="#1b1b1b", lw=0.5, label="Training block (5 × 5 km)"),
           Patch(facecolor="#5f8f5f", edgecolor="white", lw=1.5, label="Evaluation block (white)"),
           Line2D([], [], color="#777777", lw=0.9, linestyle="--", label="Four-track support")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.66, 0.02), ncol=3, frameon=False,
               fontsize=7, handlelength=1.6, columnspacing=1.0)
    lo = {k: v for k, v in RANGES.items()}
    fig.text(0.66, 0.095, f"T15, 10/22 May 2024.  R: $g_{{12}}$ [{lo['g12'][0]:.2f}, {lo['g12'][1]:.2f}]   "
             f"G: $\\gamma^0_{{VH}}$ [−{-lo['vh'][0]:.0f}, −{-lo['vh'][1]:.0f}] dB   "
             f"B: $\\gamma^0_{{VV}}-\\gamma^0_{{VH}}$ [{lo['ratio'][0]:.0f}, {lo['ratio'][1]:.0f}] dB",
             ha="center", fontsize=7)
    normalize_figure(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_overview.{ext}", **kw)
    plt.close(fig)

    inputs = [CUBE_A, CUBE_B, META_B, SPLIT_A, SPLIT_B, REF_A, NE] + sorted({v[2] for v in fps.values()})
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
           "inputs_sha256": {str(p): sha(p) for p in inputs},
           "scenes": {f"site{s}_T{t}": v[1] for (s, t), v in fps.items()},
           "support_pixels": {"site_A_four_track_valid": int(va.sum()), "site_B_four_track_valid": int(vb.sum())},
           "outputs_sha256": {str(OUT / f"fig_overview.{e}"): sha(OUT / f"fig_overview.{e}") for e in ("pdf", "png")}}
    (HERE / "fig_overview_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man["support_pixels"]), json.dumps(man["scenes"]))


if __name__ == "__main__":
    main()
