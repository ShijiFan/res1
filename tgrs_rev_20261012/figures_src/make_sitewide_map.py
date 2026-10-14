"""Site-wide held-out prediction maps, Study 1 (see ../plan/PROTOCOL_FIGREV_20261012.md, appendix).

RF, source T15 -> target T37, seed 17, I28 vs I28_G29, every evaluation block at Site A (12) and Site B (17).
Saved models only (no fitting). Features from the unchanged R1 extractor with the experiment's inputs.
Every saved sampled prediction for target T37 must be reproduced exactly, otherwise the script stops.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

from figure_typography import normalize_figure

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
OUT = HERE.parent / "manuscript" / "figures"
RUN = HERE / "sitewide_run_20261012"
RUN.mkdir(exist_ok=True)
R1_SRC = SAR / "res1_clean" / "src" / "run_R1_buffered_cv_retrain.py"
sys.path.insert(0, str(R1_SRC.parent))
import run_R1_buffered_cv_retrain as r1  # noqa: E402

SRC, TGT, SEED = "t15", "t37", 17
REPS = ("I28", "I28_G29")
COLORS = ["#287B57", "#A6BA46", "#E4A13D", "#9A4B8C", "#3B75AE"]
LABELS = ["Forest", "Grassland", "Cropland", "Built-up", "Water"]
STATUS = ["#EEEEEE", "#00807A", "#E8710A", "#A9A9A9"]
STATUS_LABELS = ["Both correct", "Corrected by coherence", "Lost with coherence", "Both wrong"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def site_a():
    A0 = SAR / "A0_gamma0_rerun_20260929"
    sv = SAR / "sar_v2"
    paths = {"cube": A0 / "data" / "cube_4track_v7.npz", "v4": sv / "data" / "flevoland_datacube_v4_multibaseline.npz",
             "cm": sv / "labels" / "consensus_mask.npz", "mmu": sv / "mmu" / "mmu_mask.npz",
             "omega": SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy",
             "split": A0 / "ws" / "splits" / "split_manifest.json"}
    c = dict(np.load(paths["cube"]))
    v4 = np.load(paths["v4"])
    cm = np.load(paths["cm"])
    mv = np.load(paths["mmu"])["mmu_valid"]
    om = np.load(paths["omega"])
    vi = np.where(mv)
    omv = om[vi]
    y_wc = v4["y"][vi].astype(int)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    gy, gx = v4["grid_y"][vi], v4["grid_x"][vi]
    split = json.loads(paths["split"].read_text())["historical_development_split"]
    runs = A0 / "ws" / "runs_R1"
    assert int(c_mask.sum()) == 287013
    return dict(name="Site A", cube=c, vi=vi, omv=omv, y=y_wc, c_mask=c_mask, gy=gy, gx=gx, split=split,
                runs=runs, shape=mv.shape, paths=paths)


def site_b():
    S1 = SAR / "site2_20260930" / "runs" / "STUDY1"
    paths = {"cube": S1 / "cube_spring.npz", "labels": S1 / "labels.npz",
             "split": S1 / "ws" / "splits" / "split_manifest.json"}
    c = dict(np.load(paths["cube"]))
    L = np.load(paths["labels"])
    vi = np.where(L["mmu_valid"])
    omv = L["omega"][vi]
    y_wc = L["y"][vi].astype(int)
    y_clc = L["clc"][vi].astype(int) - 1
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    gy, gx = L["grid_y"][vi], L["grid_x"][vi]
    split = json.loads(paths["split"].read_text())["historical_development_split"]
    assert int(c_mask.sum()) == 505806
    return dict(name="Site B", cube=c, vi=vi, omv=omv, y=y_wc, c_mask=c_mask, gy=gy, gx=gx, split=split,
                runs=S1 / "ws" / "runs_R1", shape=L["mmu_valid"].shape, paths=paths)


def predict(site):
    r1.TR = [TGT]
    feats, names = r1.extract_features(site["cube"], site["vi"])
    bids = (site["gy"] // 125) * 100 + (site["gx"] // 125)
    in_eval = np.isin(bids, site["split"]["eval_blocks"]) & site["omv"]
    idx = np.where(in_eval)[0]
    out, parity, used = {}, {}, []
    for rep in REPS:
        d = site["runs"] / f"RF_{rep}_{SRC}_s{SEED}"
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["source_track"] == SRC and cfg["seed"] == SEED and cfg["representation"] == rep
        model = joblib.load(d / "model.joblib")
        pred = np.full(len(site["y"]), -1, np.int16)
        pred[idx] = model.predict(feats[rep][TGT][idx]).astype(np.int16)
        saved = pd.read_parquet(d / "predictions.parquet")
        saved = saved[saved.target_track == TGT]
        pi = saved.pixel_index.to_numpy()
        assert np.array_equal(site["y"][pi], saved.y_true.to_numpy())
        ok = bool(np.array_equal(pred[pi], saved.y_pred.to_numpy()))
        if not ok:
            raise SystemExit(f"STOP {site['name']} {rep}: dense prediction differs from saved sample "
                             f"({int((pred[pi] != saved.y_pred.to_numpy()).sum())} of {len(pi)} rows)")
        parity[rep] = {"checked_rows": int(len(pi)), "exact": ok}
        out[rep] = pred
        used += [d / "config.json", d / "model.joblib", d / "predictions.parquet"]
    cm_ = site["c_mask"] & in_eval
    t, p0, p1 = site["y"][cm_], out["I28"][cm_], out["I28_G29"][cm_]
    bu = t == 3
    metrics = {"eval_blocks": len(site["split"]["eval_blocks"]), "omega_pixels_predicted": int(in_eval.sum()),
               "c2_pixels_evaluated": int(cm_.sum()), "c2_builtup_pixels": int(bu.sum()),
               "builtup_correct_I28": int((p0[bu] == 3).sum()), "builtup_correct_I28_G29": int((p1[bu] == 3).sum()),
               "oa_I28": float((p0 == t).mean()), "oa_I28_G29": float((p1 == t).mean()),
               "corrected": int(((p0 != t) & (p1 == t)).sum()), "lost": int(((p0 == t) & (p1 != t)).sum()),
               "c2_by_class": np.bincount(t, minlength=5).tolist(),
               "corrected_by_true_class": np.bincount(t[(p0 != t) & (p1 == t)], minlength=5).tolist(),
               "lost_by_true_class": np.bincount(t[(p0 == t) & (p1 != t)], minlength=5).tolist(),
               "class_order": LABELS, "parity": parity}
    return out, in_eval, metrics, used


def to_grid(site, vec, fill=-1):
    g = np.full(site["shape"], fill, np.int16)
    g[site["gy"], site["gx"]] = vec
    return g


def main():
    results = {}
    inputs = [R1_SRC]
    for site in (site_a(), site_b()):
        pred, in_eval, metrics, used = predict(site)
        inputs += list(site["paths"].values()) + used
        y_grid = to_grid(site, np.where(site["c_mask"] & in_eval, site["y"], -1))
        p0 = to_grid(site, np.where(in_eval, pred["I28"], -1))
        p1 = to_grid(site, np.where(in_eval, pred["I28_G29"], -1))
        ev = to_grid(site, in_eval.astype(np.int16), 0).astype(bool)
        tr_vec = np.isin((site["gy"] // 125) * 100 + (site["gx"] // 125), site["split"]["train_blocks"]) & site["omv"]
        tr = to_grid(site, tr_vec.astype(np.int16), 0).astype(bool)
        np.savez_compressed(RUN / f"{site['name'].replace(' ', '_')}_maps.npz", y=y_grid, p_I28=p0, p_I28_G29=p1,
                            eval_support=ev, train_support=tr)
        results[site["name"]] = dict(y=y_grid, p0=p0, p1=p1, ev=ev, tr=tr, split=site["split"], metrics=metrics)
        print(site["name"], json.dumps(metrics))
    plot(results)
    outs = [OUT / f"fig_sitewide_map.{e}" for e in ("pdf", "png")] + sorted(RUN.glob("*_maps.npz"))
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
           "selection_rule": "RF, source T15, target T37, seed 17; all evaluation blocks; fixed before inference",
           "inputs_sha256": {str(p): sha(p) for p in dict.fromkeys(inputs)},
           "outputs_sha256": {str(p): sha(p) for p in outs},
           "metrics": {k: v["metrics"] for k, v in results.items()}}
    (RUN / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")


def plot(results):
    cmap = ListedColormap(COLORS)
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    scmap = ListedColormap(STATUS)
    scmap.set_bad((1, 1, 1, 0))
    snorm = BoundaryNorm(np.arange(-0.5, 4.5), 4)
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.25),
                             gridspec_kw={"height_ratios": [1077 / 965, 999 / 871]})
    fig.subplots_adjust(left=0.035, right=0.995, top=0.945, bottom=0.135, wspace=0.05, hspace=0.13)
    titles = ["Reference (consensus)", "Intensity only, $I_{28}$", "With coherence, $I_{28}{+}g_{12}$",
              "Correctness change"]
    for row, (name, r) in enumerate(results.items()):
        y, p0, p1, ev, tr = r["y"], r["p0"], r["p1"], r["ev"], r["tr"]
        status = np.full(y.shape, -1, np.int16)
        m = y >= 0
        status[m & (p0 == y) & (p1 == y)] = 0
        status[m & (p0 != y) & (p1 == y)] = 1
        status[m & (p0 == y) & (p1 != y)] = 2
        status[m & (p0 != y) & (p1 != y)] = 3
        layers = [(np.ma.masked_less(y, 0), cmap, norm), (np.ma.masked_less(p0, 0), cmap, norm),
                  (np.ma.masked_less(p1, 0), cmap, norm), (np.ma.masked_less(status, 0), scmap, snorm)]
        h, w = y.shape
        for col, (arr, cm_, nm) in enumerate(layers):
            ax = axes[row, col]
            bg = np.where(tr, 0.84, 1.0)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=(0, w * 0.04, h * 0.04, 0))
            ax.imshow(arr, cmap=cm_, norm=nm, interpolation="nearest", extent=(0, w * 0.04, h * 0.04, 0))
            for b in r["split"]["eval_blocks"]:
                br, bc = divmod(b, 100)
                ax.add_patch(Rectangle((bc * 5, br * 5), 5, 5, fill=False, edgecolor="#333333", lw=0.35))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(name, fontsize=8.5)
            ax.text(0.015, 0.985, f"({chr(97 + row * 4 + col)})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85})
            if row == 1 and col == 0:
                ax.plot([w * 0.04 - 12.5, w * 0.04 - 2.5], [h * 0.04 - 2.0] * 2, color="black", lw=2,
                        solid_capstyle="butt")
                ax.text(w * 0.04 - 7.5, h * 0.04 - 3.0, "10 km", ha="center", va="bottom", fontsize=7)
    h1 = [Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)]
    h2 = [Patch(facecolor=c, label=l) for c, l in zip(STATUS, STATUS_LABELS)]
    h3 = [Patch(facecolor="#dbdbdb", edgecolor="#999999", lw=0.4, label="Training blocks")]
    fig.legend(handles=h1 + h3, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=6, frameon=False,
               fontsize=7, handlelength=1.1, columnspacing=1.0)
    fig.legend(handles=h2, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=7,
               handlelength=1.1, columnspacing=1.0)
    for ax in axes.flat:
        ax.title.set_fontfamily("Times New Roman")
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"fig_sitewide_map.{ext}", **kw)
    plt.close(fig)


if __name__ == "__main__":
    main()
