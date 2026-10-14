"""Frozen held-out block map; no fitting or outcome-based panel selection.

See ../plan/FIG_MAP_PROTOCOL_20261002.md. Exact parity with retained sampled
predictions is required before any plotted output is accepted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
from figure_typography import normalize_figure
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
RUN = HERE / "map_run_20261002"
OUT = HERE.parent / "manuscript" / "figures"
RUN.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
ROOT = SAR / "A0_gamma0_rerun_20260929"
CUBE = ROOT / "data" / "cube_4track_v7.npz"
MMU = SAR / "SAR_v2" / "mmu" / "mmu_mask.npz"
CLC = SAR / "SAR_v2" / "labels" / "consensus_mask.npz"
OMEGA = SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy"
SPLIT = ROOT / "ws" / "splits" / "split_manifest.json"
FEATURE_CODE = SAR / "res1_clean" / "src" / "run_R1_buffered_cv_retrain.py"
TRACK, SOURCE, SEED, BLOCK = "t37", "t15", 17, 205
REPS = ("I28", "I28_G29")
MODEL_DIRS = {rep: ROOT / "ws" / "runs_R1" / f"RF_{rep}_{SOURCE}_s{SEED}" for rep in REPS}
COLORS = ["#287B57", "#A6BA46", "#E4A13D", "#9A4B8C", "#3B75AE"]
LABELS = ["Forest", "Grassland", "Cropland", "Built-up", "Water"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["historical_development_split"]
    assert BLOCK in split["eval_blocks"] and BLOCK not in split["train_blocks"]
    configs = {rep: json.loads((d / "config.json").read_text()) for rep, d in MODEL_DIRS.items()}
    assert all(v["source_track"] == SOURCE and v["seed"] == SEED for v in configs.values())
    assert configs["I28"]["train_pixels_sha256"] == configs["I28_G29"]["train_pixels_sha256"]
    r, c = divmod(BLOCK, 100)
    r0, r1, c0, c1 = r * 125, (r + 1) * 125, c * 125, (c + 1) * 125
    margin = 5
    with np.load(CUBE, allow_pickle=False) as cube, np.load(MMU, allow_pickle=False) as mmu, np.load(CLC, allow_pickle=False) as clc:
        shape = cube["y"].shape
        assert (r1, c1) <= shape
        y = cube["y"][r0:r1, c0:c1].copy()
        mv = mmu["mmu_valid"]
        clc_full = np.full(shape, -1, np.int16)
        clc_full[mv] = clc["clc_on_mmu"].astype(np.int16) - 1
        omega = np.load(OMEGA, allow_pickle=False)[r0:r1, c0:c1]
        c2 = omega & (y >= 0) & (y < 5) & (y == clc_full[r0:r1, c0:c1])
        assert int(((y == 3) & c2).sum()) == 543
        assert omega.shape == (125, 125)
        rr = slice(r0 - margin, r1 + margin)
        cc = slice(c0 - margin, c1 + margin)
        data = {k: cube[k][rr, cc].copy() for k in cube.files if k == "four_track_valid" or k.startswith(f"{TRACK}_")}
        vv = cube[f"{TRACK}_d1_vv"][r0:r1, c0:c1].copy()
        coherence = cube[f"{TRACK}_g12"][r0:r1, c0:c1].copy()
        vi_y, vi_x = np.where(mv)

    sys.path.insert(0, str(FEATURE_CODE.parent))
    import run_R1_buffered_cv_retrain as r1_code  # noqa: E402

    r1_code.TR = [TRACK]
    patch_shape = data["four_track_valid"].shape
    vi = np.where(np.ones(patch_shape, dtype=bool))
    feats, feature_names = r1_code.extract_features(data, vi)
    predicted = {}
    parity = {}
    input_paths = [CUBE, MMU, CLC, OMEGA, SPLIT, FEATURE_CODE]
    for rep, d in MODEL_DIRS.items():
        assert feature_names[rep] == configs[rep]["feature_names"]
        x = feats[rep][TRACK].reshape(*patch_shape, -1)[margin:-margin, margin:-margin, :]
        assert x.shape[:2] == (125, 125) and x.shape[2] == configs[rep]["feature_dim"]
        model = joblib.load(d / "model.joblib")
        dense = np.full((125, 125), -1, dtype=np.int16)
        dense[omega] = model.predict(x[omega]).astype(np.int16)
        saved = pd.read_parquet(d / "predictions.parquet")
        saved = saved[(saved.target_track == TRACK) & (saved.block_id == BLOCK)]
        iy = vi_y[saved.pixel_index.to_numpy()] - r0
        ix = vi_x[saved.pixel_index.to_numpy()] - c0
        assert len(saved) == 800
        assert np.all(c2[iy, ix])
        assert np.array_equal(y[iy, ix], saved.y_true.to_numpy())
        equal = np.array_equal(dense[iy, ix], saved.y_pred.to_numpy())
        if not equal:
            raise RuntimeError(f"{rep}: sampled prediction parity failed")
        parity[rep] = {"checked_rows": len(saved), "exact": equal}
        predicted[rep] = dense
        np.save(RUN / f"{rep}_target_{TRACK}_block_{BLOCK}.npy", dense)
        input_paths += [d / "config.json", d / "model.joblib", d / "predictions.parquet"]

    base, aug = (predicted[rep] for rep in REPS)
    truth = y[c2]
    p0, p1 = base[c2], aug[c2]
    cm0 = confusion_matrix(truth, p0, labels=range(5))
    cm1 = confusion_matrix(truth, p1, labels=range(5))
    metrics = {
        "block": BLOCK, "source_track": SOURCE, "target_track": TRACK, "seed": SEED,
        "n_omega": int(omega.sum()), "n_c2": int(c2.sum()),
        "n_c2_by_class": np.bincount(truth, minlength=5).tolist(),
        "oa_i28": float(np.mean(p0 == truth)), "oa_i28_g29": float(np.mean(p1 == truth)),
        "built_up_correct_i28": int(cm0[3, 3]),
        "built_up_correct_i28_g29": int(cm1[3, 3]),
        "confusion_i28": cm0.tolist(), "confusion_i28_g29": cm1.tolist(),
        "parity": parity,
    }
    np.savez_compressed(RUN / "reference_and_masks.npz", y=y, omega=omega, c2=c2)
    plot(vv, coherence, y, omega, c2, base, aug)
    outputs = [RUN / f"{rep}_target_{TRACK}_block_{BLOCK}.npy" for rep in REPS]
    outputs += [RUN / "reference_and_masks.npz"]
    outputs += [OUT / f"fig_holdout_map.{ext}" for ext in ("pdf", "svg", "png")]
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha(Path(__file__)),
                "inputs_sha256": {str(p): sha(p) for p in input_paths},
                "outputs_sha256": {str(p): sha(p) for p in outputs},
                "metrics": metrics,
                "selection_rule": "Predeclared Site A evaluation block 205, RF source T15 target T37 seed 17"}
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def plot(vv, coherence, y, omega, c2, base, aug) -> None:
    cmap_cls = ListedColormap(COLORS)
    cmap_cls.set_bad("white")
    norm_cls = BoundaryNorm(np.arange(-0.5, 5.5), cmap_cls.N)
    status = np.full(y.shape, -1, dtype=np.int8)
    both = (base == y) & (aug == y)
    recovered = (base != y) & (aug == y)
    lost = (base == y) & (aug != y)
    neither = (base != y) & (aug != y)
    status[c2 & both] = 0
    status[c2 & recovered] = 1
    status[c2 & lost] = 2
    status[c2 & neither] = 3
    status_cmap = ListedColormap(["#EEEEEE", "#00807A", "#E8710A", "#A9A9A9"])
    status_cmap.set_bad("white")
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.00))
    fig.subplots_adjust(left=0.035, right=0.99, top=0.92, bottom=0.205, wspace=0.055, hspace=0.19)
    panels = [
        (np.ma.masked_where(~omega, vv), "Greys_r", None, -18, -2, "VV backscatter"),
        (np.ma.masked_where(~omega, coherence), "cividis", None, 0, 1, "VV coherence"),
        (np.ma.masked_where(~c2, y), cmap_cls, norm_cls, None, None, "Consensus reference"),
        (np.ma.masked_where(~omega, base), cmap_cls, norm_cls, None, None, "Intensity RF"),
        (np.ma.masked_where(~omega, aug), cmap_cls, norm_cls, None, None, "Intensity + coherence RF"),
        (np.ma.masked_where(~c2, status), status_cmap, BoundaryNorm(np.arange(-0.5, 4.5), 4), None, None, "Correctness change"),
    ]
    for j, (arr, cmap, norm, vmin, vmax, title) in enumerate(panels):
        ax = axes.flat[j]
        ax.imshow(arr, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest",
                  origin="upper", extent=(0, 5, 5, 0))
        ax.set_title(f"({chr(97+j)}) {title}", loc="left", fontsize=7.5, fontweight="bold", pad=4)
        ax.set_xticks([0, 2.5, 5] if j >= 3 else [])
        ax.set_yticks([0, 2.5, 5] if j % 3 == 0 else [])
        ax.tick_params(length=2, pad=1, labelsize=6.5)
        if j >= 3:
            ax.set_xlabel("Distance (km)", fontsize=7)
        if j % 3 == 0:
            ax.set_ylabel("Distance (km)", fontsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    cls_handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)]
    status_handles = [Patch(facecolor=color, label=label) for color, label in zip(status_cmap.colors,
                      ["Both correct", "Corrected by coherence", "Lost with coherence", "Both wrong"])]
    fig.legend(handles=cls_handles, loc="lower center", bbox_to_anchor=(0.5, 0.085),
               ncol=5, frameon=False, fontsize=6.5, handlelength=1)
    fig.legend(handles=status_handles, loc="lower center", bbox_to_anchor=(0.5, 0.033),
               ncol=4, frameon=False, fontsize=6.5, handlelength=1)
    normalize_figure(fig)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_holdout_map.{ext}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
