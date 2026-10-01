"""Assemble the public reproducibility package from the working folders under E:\\research\\SAR.

Copies code, frozen protocols, result tables, per-pixel/per-parcel predictions and confusion arrays.
Skips model weights (*.joblib, *.pt), data cubes, raw SAR products, caches and backups, which are either
large or rebuildable from public sources with the included scripts. Re-running overwrites copied files.
"""
import shutil
from pathlib import Path

SAR = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
SKIP_SUFFIX = {".joblib", ".pt", ".pyc", ".bak", ".zip", ".gpkg", ".tif", ".log"}
SKIP_PARTS = {"__pycache__"}
n_files, n_bytes = 0, 0


def put(src, dst, patterns=("*",)):
    """Copy files matching patterns (recursive) from src into REPO/dst, keeping relative paths."""
    global n_files, n_bytes
    src = SAR / src
    for pat in patterns:
        for f in (src.rglob(pat) if src.is_dir() else [src]):
            if not f.is_file() or f.suffix.lower() in SKIP_SUFFIX or SKIP_PARTS & set(f.parts):
                continue
            rel = f.relative_to(src) if src.is_dir() else Path(f.name)
            out = REPO / dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n_files += 1
            n_bytes += f.stat().st_size


A = "A0_gamma0_rerun_20260929"
put(f"{A}/scripts", "site_A/study1/scripts", ("*.py", "*.sh"))
put(f"{A}/PROTOCOL.md", "site_A/study1")
put(f"{A}/RESULTS.md", "site_A/study1")
for d in ("tables", "tables_full", "tables_cnn", "autumn/tables", "configs", "cnn_checks", "splits"):
    put(f"{A}/ws/{d}", f"site_A/study1/results/{d}")
put(f"{A}/ws/bag_check", "site_A/study1/results/bag_check", ("*.csv", "*.json", "*.npz"))
put(f"{A}/ws/provenance.json", "site_A/study1/results")
put(f"{A}/data", "site_A/study1/results/cube_meta", ("*_meta.json",))
put(f"{A}/labels", "site_A/study1/results/labels_meta", ("*.json",))
put(f"{A}/ws/runs_R1", "site_A/study1/predictions/runs_R1", ("predictions.parquet", "blocks_confusion.npy", "config.json"))
put(f"{A}/ws/runs_full", "site_A/study1/predictions/runs_full", ("blocks_confusion.npy", "*.json"))
put(f"{A}/ws/runs_cnn", "site_A/study1/predictions/runs_cnn", ("blocks_confusion*.npy", "config.json"))
put(f"{A}/ws/runs_cnn2", "site_A/study1/predictions/runs_cnn2", ("blocks_confusion*.npy", "config.json"))
put(f"{A}/ws/runs_autumn", "site_A/study1/predictions/runs_autumn", ("blocks_confusion.npy",))
put(f"{A}/ws/autumn/runs_ext", "site_A/study1/predictions/autumn_runs_ext", ("predictions.parquet",))

B1 = "A1_observation_budget_20260923"
M1 = f"{B1}/runs/20260924_A1_MAIN"
put(f"{B1}/scripts", "site_A/study2/scripts", ("*.py",))
for f in ("A1_PREREG.md", "A1_PREREG.sha256", "E0_QC.md", "R2_CHECK.txt"):
    put(f"{M1}/reports/{f}", "site_A/study2/prereg_and_qc")
put(f"{M1}/stats", "site_A/study2/results/stats")
put(f"{M1}/stats_r2", "site_A/study2/results/stats_r2")
put(f"{M1}/experiments", "site_A/study2/predictions", ("REGISTRY.csv", "*.npz", "parcels_*.csv"))
put(f"{M1}/cube", "site_A/study2/results/cube_index", ("PRODUCT_INDEX.csv", "E0_SUMMARY.json"))
put(f"{B1}/runs/20260923_G0L_COMMON_GRID", "site_A/study2/grid_labels_splits", ("*.json", "*.csv", "*.md"))
put(f"{B1}/runs/20260923_G0R_CATALOG_REPAIR/manifests", "site_A/study2/manifests", ("*.csv",))

G = "site2_G0_20260930"
put(G, "site_B/G0_and_prereg", ("*.md", "*.sha256", "*.py", "*.json", "*.csv"))
S = "site2_20260930"
put(f"{S}/scripts", "site_B/scripts", ("*.py", "*.sh"))
put(f"{S}/manifests", "site_B/manifests")
for f in ("RESULTS_SITE2.md", "RESULTS_SITE2_STUDY2_CRITERIA.json"):
    put(f"{S}/{f}", "site_B")
put(f"{S}/runs/MAIN/stats", "site_B/study2/results/stats")
put(f"{S}/runs/MAIN/stats_r2", "site_B/study2/results/stats_r2")
put(f"{S}/runs/MAIN/reports", "site_B/study2/prereg_and_qc", ("*.md", "*.sha256", "*.json", "*.txt"))
put(f"{S}/runs/MAIN/experiments", "site_B/study2/predictions", ("REGISTRY.csv", "*.npz", "parcels_*.csv"))
put(f"{S}/runs/MAIN/cube", "site_B/study2/results/cube_index", ("PRODUCT_INDEX.csv", "E0_SUMMARY.json", "product_index.jsonl"))
put(f"{S}/runs/G0L_COMMON_GRID", "site_B/study2/grid_labels_splits", ("*.json", "*.csv", "*.md"))
for f in ("meta.json", "shift_test.json", "split_manifest.json"):
    put(f"{S}/runs/STUDY1/{f}", "site_B/study1/results")
for d in ("tables", "configs", "cnn_checks"):
    put(f"{S}/runs/STUDY1/ws/{d}", f"site_B/study1/results/{d}")
put(f"{S}/runs/STUDY1/ws/bag_check", "site_B/study1/results/bag_check", ("*.csv", "*.json", "*.npz"))
put(f"{S}/runs/STUDY1/ws/provenance.json", "site_B/study1/results")
put(f"{S}/runs/STUDY1/ws/runs_R1", "site_B/study1/predictions/runs_R1", ("predictions.parquet", "blocks_confusion.npy", "config.json"))
put(f"{S}/runs/STUDY1/ws/runs_cnn2", "site_B/study1/predictions/runs_cnn2", ("blocks_confusion*.npy", "config.json"))
put(f"{S}/runs/STUDY1/ws/runs_autumn", "site_B/study1/predictions/runs_autumn", ("blocks_confusion.npy",))
put(f"{S}/labels", "site_B/labels_meta", ("*meta*.json",))

T = "temporal_dl_20261001"
put(T, "temporal_dl", ("*.py", "*.sh", "PROTOCOL.md", "PROTOCOL.sha256", "RESULTS.md"))
for s in ("site1", "site2"):
    put(f"{T}/{s}", f"temporal_dl/{s}", ("REGISTRY.csv", "cells.csv", "judgement.json", "*.npz"))

put("merged_tgrs_20260930/figures_src", "figures", ("make_figures.py", "site2_cnn_contrast.json"))
print(f"copied {n_files} files, {n_bytes / 1e6:.1f} MB")
