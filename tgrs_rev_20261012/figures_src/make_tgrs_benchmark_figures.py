"""Two original figures informed by the visual grammar of published TGRS work.

The split map uses frozen Study 1 split manifests. The four-track montage uses
the current gamma0 Site A cube and the ROI fixed for fig_detail. No model is
trained and no outcome-based crop is selected here.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SITE_A = SAR / "A0_gamma0_rerun_20260929" / "data" / "cube_4track_v7.npz"
SPLIT_A = SAR / "A0_gamma0_rerun_20260929" / "ws" / "splits" / "split_manifest.json"
SPLIT_B = SAR / "site2_20260930" / "runs" / "STUDY1" / "ws" / "splits" / "split_manifest.json"
ROI_MANIFEST = HERE / "visual_additions_manifest.json"

INK = "#21303A"
TRAIN = "#39779E"
EVAL = "#DA7A3E"
EMPTY = "#F4F6F8"
FOLDS = ["#4A83A6", "#5F9D85", "#8D7BAD"]

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.edgecolor": INK, "axes.linewidth": 0.5,
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save(fig: plt.Figure, stem: str) -> None:
    normalize_figure(fig)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def split_panel(ax, manifest: dict, site: str, fold_view: bool = False) -> dict:
    split = manifest["historical_development_split"]
    train = set(split["train_blocks"])
    evaluation = set(split["eval_blocks"])
    if train & evaluation:
        raise ValueError(f"{site}: training and evaluation blocks overlap")
    all_blocks = train | evaluation
    if fold_view:
        folds = manifest["source_spatial_cv_3fold"]
        fold_sets = [set(folds[f"fold_{j}_blocks"]) for j in (1, 2, 3)]
        if set.union(*fold_sets) != train or any(fold_sets[i] & fold_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Site A spatial folds do not partition training blocks")
    nrows = max(b // 100 for b in all_blocks) + 1
    ncols = max(b % 100 for b in all_blocks) + 1
    for bid in sorted(all_blocks):
        row, col = divmod(bid, 100)
        color = EVAL if bid in evaluation else TRAIN
        if fold_view:
            color = EMPTY if bid in evaluation else FOLDS[next(i for i, group in enumerate(fold_sets) if bid in group)]
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=0.8))
        ax.text(col + 0.5, row + 0.5, str(bid), ha="center", va="center",
                fontsize=5.4, color=INK if (fold_view and bid in evaluation) else "white")
    ax.set_xlim(-0.12, ncols + 0.12)
    ax.set_ylim(nrows + 0.12, -0.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{site}: spatial folds" if fold_view else f"{site}: train / evaluation",
                 fontweight="bold", loc="left", pad=6)
    return {"train_blocks": len(train), "eval_blocks": len(evaluation),
            "nrows": nrows, "ncols": ncols}


def fig_spatial_protocol() -> dict:
    a = json.loads(SPLIT_A.read_text(encoding="utf-8"))
    b = json.loads(SPLIT_B.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.13))
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.145, top=0.88, wspace=0.10)
    panels = [(a, "(a) Site A", False), (b, "(b) Site B", False), (a, "(c) Site A", True)]
    info = {}
    for ax, (m, title, folds) in zip(axes, panels):
        info[title] = split_panel(ax, m, title, folds)
    handles = [Patch(facecolor=TRAIN, label="Training"), Patch(facecolor=EVAL, label="Evaluation")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.015),
               ncol=2, frameon=False, fontsize=7)
    handles_folds = [Patch(facecolor=c, label=f"Fold {j}") for j, c in enumerate(FOLDS, 1)]
    handles_folds.append(Patch(facecolor=EMPTY, edgecolor="#CAD0D6", label="Evaluation"))
    fig.legend(handles=handles_folds, loc="lower right", bbox_to_anchor=(0.97, 0.015),
               ncol=4, frameon=False, fontsize=7)
    save(fig, "fig_spatial_protocol")
    return info


def fig_four_track() -> dict:
    roi = json.loads(ROI_MANIFEST.read_text(encoding="utf-8"))["rois"]["Site A"]
    r0, r1 = roi["crop_rows"]
    c0, c1 = roi["crop_cols"]
    sl = np.s_[r0:r1, c0:c1]
    tracks = ("t15", "t37", "t88", "t139")
    dates = ("10–22 May", "12–24 May", "15–27 May", "7–19 May")
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
    fig.subplots_adjust(left=0.042, right=0.992, bottom=0.24, top=0.90,
                        wspace=0.06, hspace=0.23)
    with np.load(SITE_A, allow_pickle=False) as cube:
        valid_common = cube["mutual_valid"][sl]
        for col, (track, datestr) in enumerate(zip(tracks, dates)):
            valid = valid_common & cube[f"{track}_valid"][sl]
            vv = np.ma.masked_where(~valid, cube[f"{track}_d1_vv"][sl])
            coherence = np.ma.masked_where(~valid, cube[f"{track}_g12"][sl])
            im_vv = axes[0, col].imshow(vv, origin="upper", extent=(0, 6.4, 6.4, 0),
                                        cmap="Greys_r", vmin=-18, vmax=-2, interpolation="nearest")
            im_coh = axes[1, col].imshow(coherence, origin="upper", extent=(0, 6.4, 6.4, 0),
                                         cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
            axes[0, col].set_title(f"{track.upper()} · {datestr}", fontsize=7.4, pad=4)
            for row in (0, 1):
                ax = axes[row, col]
                ax.set_xticks([0, 3, 6] if row == 1 else [])
                ax.set_yticks([0, 3, 6] if col == 0 else [])
                ax.tick_params(length=2, pad=1)
                ax.set_xlabel("Distance (km)" if row == 1 else "")
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
            if col == 0:
                axes[0, col].set_ylabel("VV backscatter\nDistance (km)")
                axes[1, col].set_ylabel("VV coherence\nDistance (km)")
            axes[0, col].text(0.04, 0.06, f"({chr(97 + col)})", transform=axes[0, col].transAxes,
                              fontsize=8, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            axes[1, col].text(0.04, 0.06, f"({chr(101 + col)})", transform=axes[1, col].transAxes,
                              fontsize=8, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    c1ax = fig.add_axes([0.13, 0.115, 0.30, 0.013])
    c2ax = fig.add_axes([0.58, 0.115, 0.30, 0.013])
    fig.colorbar(im_vv, cax=c1ax, orientation="horizontal", ticks=[-18, -10, -2]).set_label("$\\gamma^0_{VV}$ (dB)", fontsize=7)
    fig.colorbar(im_coh, cax=c2ax, orientation="horizontal", ticks=[0, 0.5, 1]).set_label("$g_{12}$", fontsize=7)
    save(fig, "fig_four_track")
    return {"crop_rows": [r0, r1], "crop_cols": [c0, c1], "tracks": list(tracks),
            "crop_width_km": 6.4}


def main() -> None:
    splits = fig_spatial_protocol()
    roi = fig_four_track()
    outputs = [OUT / f"{stem}.{ext}" for stem in ("fig_spatial_protocol", "fig_four_track")
               for ext in ("pdf", "svg", "png")]
    inputs = [SITE_A, SPLIT_A, SPLIT_B, ROI_MANIFEST]
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha256(Path(__file__)),
                "inputs_sha256": {str(p): sha256(p) for p in inputs},
                "outputs_sha256": {str(p): sha256(p) for p in outputs},
                "splits": splits, "roi": roi,
                "note": "Visualization only; no new estimates or dense prediction maps."}
    (HERE / "tgrs_benchmark_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
