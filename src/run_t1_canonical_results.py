"""
Phase T1: Canonical Results Generation from runs_R1 (400m Buffered Protocol).
Includes:
- T1-A: Raw complete test-set point estimates (cross & in, BA & OA)
- T1-B: 20,000 paired block bootstrap for 5 contrast sets
- T1-C: Class-specific recall & pooled confusion matrix (with unique & total sample size)
- T1-D: Full 4x4 transfer matrices
- Output files: T1_canonical_results.json, T1_main_results.csv, T1_contrasts.csv,
  T1_per_class.csv, T1_transfer_matrix.csv, T1_class_dropout.csv, T1_old_vs_new.csv
- Strict acceptance assertions.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE = Path(r"E:\research\SAR\tgrs_final_campaign_20260915")
RUNS_OLD = WORKSPACE / "runs"
RUNS_NEW = WORKSPACE / "runs_R1"
TABLES_DIR = WORKSPACE / "tables"
TABLES_DIR.mkdir(exist_ok=True)

TR = ["t15", "t37", "t88", "t139"]
REPS = ["I8", "I28", "I8_G9", "F12", "I28_G29"]
SEEDS = [17, 29, 43]
CLASSES = ["Forest", "Grassland", "Cropland", "Urban", "Water"]

print("=== Starting T1: Canonical Results from runs_R1 ===")

# --- Step 1: Load all 120 runs from runs_R1 and runs_OLD ---
conf_new = {l: {r: {s: {} for s in TR} for r in REPS} for l in ["RF", "SVC"]}
conf_old = {l: {r: {s: {} for s in TR} for r in REPS} for l in ["RF", "SVC"]}

for l in ["RF", "SVC"]:
    for r in REPS:
        for s in TR:
            for seed in SEEDS:
                p_new = RUNS_NEW / f"{l}_{r}_{s}_s{seed}" / "blocks_confusion.npy"
                p_old = RUNS_OLD / f"{l}_{r}_{s}_s{seed}" / "blocks_confusion.npy"
                conf_new[l][r][s][seed] = np.load(p_new)
                conf_old[l][r][s][seed] = np.load(p_old)

print("Loaded all 120 runs from runs_R1 and 120 runs from runs (0 missing).")

# --- Step 2: Helper Functions for Exact Metrics ---
def compute_ba_from_cm(cm_5x5):
    """5-class macro-average recall (fixed denominator)."""
    supp = cm_5x5.sum(axis=1)
    rec = np.divide(np.diag(cm_5x5), supp, out=np.zeros(5, dtype=float), where=supp > 0)
    return float(rec.mean() * 100.0)

def compute_oa_from_cm(cm_5x5):
    """Overall accuracy."""
    tot = cm_5x5.sum()
    if tot > 0:
        return float(np.diag(cm_5x5).sum() / tot * 100.0)
    return 0.0

def compute_cell_metrics(conf_dict):
    seed_records = []
    cell_matrices = {}
    
    for l in ["RF", "SVC"]:
        for r in REPS:
            for seed in SEEDS:
                ba_mat = np.zeros((4, 4))
                oa_mat = np.zeros((4, 4))
                for si, src in enumerate(TR):
                    cm_all = conf_dict[l][r][src][seed].sum(axis=1) # (4, 5, 5)
                    for ti in range(4):
                        c = cm_all[ti]
                        ba_mat[si, ti] = compute_ba_from_cm(c)
                        oa_mat[si, ti] = compute_oa_from_cm(c)
                
                cross_mask = ~np.eye(4, dtype=bool)
                in_mask = np.eye(4, dtype=bool)
                
                cell_matrices[(l, r, seed, "ba")] = ba_mat
                cell_matrices[(l, r, seed, "oa")] = oa_mat
                
                seed_records.append({
                    "learner": l, "representation": r, "seed": seed,
                    "cross_ba": ba_mat[cross_mask].mean(),
                    "in_ba": ba_mat[in_mask].mean(),
                    "cross_oa": oa_mat[cross_mask].mean(),
                    "in_oa": oa_mat[in_mask].mean(),
                })
                
    df_seeds = pd.DataFrame(seed_records)
    df_mean = df_seeds.groupby(["learner", "representation"]).agg({
        "cross_ba": ["mean", "std"],
        "in_ba": ["mean", "std"],
        "cross_oa": ["mean", "std"],
        "in_oa": ["mean", "std"],
    }).reset_index()
    
    df_mean.columns = [
        "learner", "representation",
        "cross_ba_mean", "cross_ba_std",
        "in_ba_mean", "in_ba_std",
        "cross_oa_mean", "cross_oa_std",
        "in_oa_mean", "in_oa_std",
    ]
    return df_seeds, df_mean, cell_matrices

df_seeds_new, df_main_new, cell_mats_new = compute_cell_metrics(conf_new)
df_seeds_old, df_main_old, cell_mats_old = compute_cell_metrics(conf_old)

df_main_new.round(4).to_csv(TABLES_DIR / "T1_main_results.csv", index=False)
print("Saved tables/T1_main_results.csv")

# --- Step 3: T1-B Paired Block Bootstrap (20,000 Replicates) ---
N_BOOT = 20000
rng = np.random.RandomState(42)
boot_indices = rng.randint(0, 12, size=(N_BOOT, 12))

pooled_a_new = {}
for l in ["RF", "SVC"]:
    for r in REPS:
        arr = np.zeros((4, 4, 12, 5, 5), dtype=np.float64)
        for si, src in enumerate(TR):
            for seed in SEEDS:
                arr[si] += conf_new[l][r][src][seed]
        arr /= len(SEEDS)
        pooled_a_new[(l, r)] = arr

counts = np.eye(12, dtype=np.float64)[boot_indices].sum(axis=1)

sample_arr = pooled_a_new[("RF", "I28")]
cm_boot_check = np.einsum("nb,stbkj->nstkj", counts, sample_arr, optimize=True)
den_check = cm_boot_check.sum(axis=-1)
cross_mask = ~np.eye(4, dtype=bool)
valid_mask = (den_check[:, cross_mask, :] > 0).all(axis=(1, 2))

dropout_records = []
for c_idx, c_name in enumerate(CLASSES):
    has_zero = (den_check[:, cross_mask, c_idx] == 0).any(axis=1)
    dropout_records.append({
        "class_id": c_idx,
        "class_name": c_name,
        "zero_replicates": int(has_zero.sum()),
        "total_replicates": N_BOOT,
        "dropout_rate": float(has_zero.sum() / N_BOOT),
    })
df_drop = pd.DataFrame(dropout_records)
df_drop.to_csv(TABLES_DIR / "T1_class_dropout.csv", index=False)
print("Saved tables/T1_class_dropout.csv")

boots_new = {}
in_mask = np.eye(4, dtype=bool)

for (l, r), a in pooled_a_new.items():
    cm_boot = np.einsum("nb,stbkj->nstkj", counts, a, optimize=True)
    den = cm_boot.sum(axis=-1)
    rec = np.divide(np.diagonal(cm_boot, axis1=-2, axis2=-1), den, out=np.zeros_like(den), where=den > 0)
    ba_boot = rec.mean(axis=-1) * 100.0
    oa_boot = np.diagonal(cm_boot, axis1=-2, axis2=-1).sum(axis=-1) / cm_boot.sum(axis=(-2, -1)) * 100.0
    
    boots_new[(l, r, "cross_ba")] = ba_boot[:, cross_mask].mean(axis=1)
    boots_new[(l, r, "in_ba")] = ba_boot[:, in_mask].mean(axis=1)
    boots_new[(l, r, "cross_oa")] = oa_boot[:, cross_mask].mean(axis=1)
    boots_new[(l, r, "in_oa")] = oa_boot[:, in_mask].mean(axis=1)

contrast_records = []

def record_contrast(cid, learner, metric, pt_val, boot_series):
    vals = boot_series[valid_mask]
    ci_lo = float(np.percentile(vals, 2.5))
    ci_hi = float(np.percentile(vals, 97.5))
    excludes_zero = bool((ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0))
    contrast_records.append({
        "contrast_id": cid,
        "learner": learner,
        "metric": metric,
        "point_estimate": round(pt_val, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "excludes_zero": excludes_zero,
        "n_boot": N_BOOT,
        "n_valid_boot": int(valid_mask.sum()),
    })

for l in ["RF", "SVC"]:
    # C-main: I28_G29 - I28
    pt_main_ba = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28_G29")]["cross_ba_mean"].iloc[0] - \
                       df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28")]["cross_ba_mean"].iloc[0])
    boot_main_ba = boots_new[(l, "I28_G29", "cross_ba")] - boots_new[(l, "I28", "cross_ba")]
    record_contrast("C-main", l, "cross_ba", pt_main_ba, boot_main_ba)

    pt_main_oa = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28_G29")]["cross_oa_mean"].iloc[0] - \
                       df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28")]["cross_oa_mean"].iloc[0])
    boot_main_oa = boots_new[(l, "I28_G29", "cross_oa")] - boots_new[(l, "I28", "cross_oa")]
    record_contrast("C-main", l, "cross_oa", pt_main_oa, boot_main_oa)

    # C-paired: Delta_cross - Delta_in
    pt_in_ba = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28_G29")]["in_ba_mean"].iloc[0] - \
                     df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28")]["in_ba_mean"].iloc[0])
    pt_paired_ba = pt_main_ba - pt_in_ba
    boot_in_ba = boots_new[(l, "I28_G29", "in_ba")] - boots_new[(l, "I28", "in_ba")]
    boot_paired_ba = boot_main_ba - boot_in_ba
    record_contrast("C-paired", l, "delta_cross_minus_delta_in_ba", pt_paired_ba, boot_paired_ba)

    pt_in_oa = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28_G29")]["in_oa_mean"].iloc[0] - \
                     df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28")]["in_oa_mean"].iloc[0])
    pt_paired_oa = pt_main_oa - pt_in_oa
    boot_in_oa = boots_new[(l, "I28_G29", "in_oa")] - boots_new[(l, "I28", "in_oa")]
    boot_paired_oa = boot_main_oa - boot_in_oa
    record_contrast("C-paired", l, "delta_cross_minus_delta_in_oa", pt_paired_oa, boot_paired_oa)

    # C-ablation: F12 - I8_G9
    pt_abl_ba = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "F12")]["cross_ba_mean"].iloc[0] - \
                      df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I8_G9")]["cross_ba_mean"].iloc[0])
    boot_abl_ba = boots_new[(l, "F12", "cross_ba")] - boots_new[(l, "I8_G9", "cross_ba")]
    record_contrast("C-ablation", l, "cross_ba", pt_abl_ba, boot_abl_ba)

    # C-scalar: I8_G9 - I8
    pt_scal_ba = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I8_G9")]["cross_ba_mean"].iloc[0] - \
                       df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I8")]["cross_ba_mean"].iloc[0])
    boot_scal_ba = boots_new[(l, "I8_G9", "cross_ba")] - boots_new[(l, "I8", "cross_ba")]
    record_contrast("C-scalar", l, "cross_ba", pt_scal_ba, boot_scal_ba)

    # C-baseline: I28 - I8
    pt_base_ba = float(df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I28")]["cross_ba_mean"].iloc[0] - \
                       df_main_new[(df_main_new.learner == l) & (df_main_new.representation == "I8")]["cross_ba_mean"].iloc[0])
    boot_base_ba = boots_new[(l, "I28", "cross_ba")] - boots_new[(l, "I8", "cross_ba")]
    record_contrast("C-baseline", l, "cross_ba", pt_base_ba, boot_base_ba)

df_contrasts = pd.DataFrame(contrast_records)
df_contrasts.to_csv(TABLES_DIR / "T1_contrasts.csv", index=False)
print("Saved tables/T1_contrasts.csv")

# --- Step 4: T1-C Per-Class Recall & Confusion Matrix ---
pred_sample = pd.read_parquet(RUNS_NEW / "RF_I28_t15_s17" / "predictions.parquet")
eval_unique = pred_sample[pred_sample.target_track == "t15"][["pixel_index", "y_true"]].drop_duplicates()
unique_pixel_counts = eval_unique["y_true"].value_counts().sort_index().to_dict()

per_class_records = []

for l in ["RF", "SVC"]:
    cm_i28_pooled = np.zeros((5, 5), dtype=np.float64)
    cm_g29_pooled = np.zeros((5, 5), dtype=np.float64)
    
    for s in SEEDS:
        for si, src in enumerate(TR):
            for ti in range(4):
                if si != ti:
                    cm_i28_pooled += conf_new[l]["I28"][src][s][ti].sum(axis=0)
                    cm_g29_pooled += conf_new[l]["I28_G29"][src][s][ti].sum(axis=0)
                    
    rec_i28 = np.diag(cm_i28_pooled) / cm_i28_pooled.sum(axis=1) * 100.0
    rec_g29 = np.diag(cm_g29_pooled) / cm_g29_pooled.sum(axis=1) * 100.0
    
    norm_cm_i28 = (cm_i28_pooled / cm_i28_pooled.sum(axis=1, keepdims=True) * 100.0).round(2)
    norm_cm_g29 = (cm_g29_pooled / cm_g29_pooled.sum(axis=1, keepdims=True) * 100.0).round(2)
    
    for c_idx, c_name in enumerate(CLASSES):
        u_px = unique_pixel_counts.get(c_idx, 0)
        tot_pred = int(cm_i28_pooled.sum(axis=1)[c_idx])
        d_rec = rec_g29[c_idx] - rec_i28[c_idx]
        ba_contrib = d_rec / 5.0
        
        per_class_records.append({
            "learner": l,
            "class_id": c_idx,
            "class_name": c_name,
            "unique_eval_pixels": u_px,
            "cumulative_predictions": tot_pred,
            "recall_I28": round(rec_i28[c_idx], 2),
            "recall_I28_G29": round(rec_g29[c_idx], 2),
            "delta_recall": round(d_rec, 2),
            "ba_contribution_pp": round(ba_contrib, 2),
            "conf_to_forest_I28": norm_cm_i28[c_idx, 0],
            "conf_to_forest_I28_G29": norm_cm_g29[c_idx, 0],
            "conf_to_urban_I28": norm_cm_i28[c_idx, 3],
            "conf_to_urban_I28_G29": norm_cm_g29[c_idx, 3],
        })

df_per_class = pd.DataFrame(per_class_records)
df_per_class.to_csv(TABLES_DIR / "T1_per_class.csv", index=False)
print("Saved tables/T1_per_class.csv")

# --- Step 5: T1-D Full 4x4 Transfer Matrices ---
transfer_records = []
for l in ["RF", "SVC"]:
    for r in ["I28", "I28_G29"]:
        ba_grid = np.mean([cell_mats_new[(l, r, s, "ba")] for s in SEEDS], axis=0)
        oa_grid = np.mean([cell_mats_new[(l, r, s, "oa")] for s in SEEDS], axis=0)
        
        for si, src in enumerate(TR):
            for ti, tgt in enumerate(TR):
                transfer_records.append({
                    "learner": l,
                    "representation": r,
                    "source_track": src,
                    "target_track": tgt,
                    "is_in_track": (si == ti),
                    "ba": round(ba_grid[si, ti], 2),
                    "oa": round(oa_grid[si, ti], 2),
                })

df_transfer = pd.DataFrame(transfer_records)
df_transfer.to_csv(TABLES_DIR / "T1_transfer_matrix.csv", index=False)
print("Saved tables/T1_transfer_matrix.csv")

# --- Step 6: Old vs New Comparison Table (T1_old_vs_new.csv) ---
old_vs_new_records = []
for l in ["RF", "SVC"]:
    for m in ["ba", "oa"]:
        c29_o = df_seeds_old[(df_seeds_old.learner == l) & (df_seeds_old.representation == "I28_G29")][f"cross_{m}"].values
        c28_o = df_seeds_old[(df_seeds_old.learner == l) & (df_seeds_old.representation == "I28")][f"cross_{m}"].values
        i29_o = df_seeds_old[(df_seeds_old.learner == l) & (df_seeds_old.representation == "I28_G29")][f"in_{m}"].values
        i28_o = df_seeds_old[(df_seeds_old.learner == l) & (df_seeds_old.representation == "I28")][f"in_{m}"].values
        
        d_cross_o = c29_o - c28_o
        d_in_o = i29_o - i28_o
        paired_o = d_cross_o - d_in_o
        
        c29_n = df_seeds_new[(df_seeds_new.learner == l) & (df_seeds_new.representation == "I28_G29")][f"cross_{m}"].values
        c28_n = df_seeds_new[(df_seeds_new.learner == l) & (df_seeds_new.representation == "I28")][f"cross_{m}"].values
        i29_n = df_seeds_new[(df_seeds_new.learner == l) & (df_seeds_new.representation == "I28_G29")][f"in_{m}"].values
        i28_n = df_seeds_new[(df_seeds_new.learner == l) & (df_seeds_new.representation == "I28")][f"in_{m}"].values
        
        d_cross_n = c29_n - c28_n
        d_in_n = i29_n - i28_n
        paired_n = d_cross_n - d_in_n
        
        old_vs_new_records.append({
            "learner": l,
            "metric": m.upper(),
            "old_cross_baseline": round(c28_o.mean(), 4),
            "new_cross_baseline": round(c28_n.mean(), 4),
            "old_cross_augmented": round(c29_o.mean(), 4),
            "new_cross_augmented": round(c29_n.mean(), 4),
            "old_delta_cross": round(d_cross_o.mean(), 4),
            "new_delta_cross": round(d_cross_n.mean(), 4),
            "old_in_baseline": round(i28_o.mean(), 4),
            "new_in_baseline": round(i28_n.mean(), 4),
            "old_in_augmented": round(i29_o.mean(), 4),
            "new_in_augmented": round(i29_n.mean(), 4),
            "old_delta_in": round(d_in_o.mean(), 4),
            "new_delta_in": round(d_in_n.mean(), 4),
            "old_delta_cross_minus_delta_in": round(paired_o.mean(), 4),
            "new_delta_cross_minus_delta_in": round(paired_n.mean(), 4),
        })

df_ovn = pd.DataFrame(old_vs_new_records)
df_ovn.to_csv(TABLES_DIR / "T1_old_vs_new.csv", index=False)
print("Saved tables/T1_old_vs_new.csv")
print(df_ovn.to_string(index=False))

# --- Step 7: Single Canonical JSON (tables/T1_canonical_results.json) ---
canonical_data = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "protocol": "400m_buffered_spatial_cv_and_test",
    "primary_source_directory": "runs_R1",
    "historical_baseline_directory": "runs",
    "main_results_summary": df_main_new.to_dict(orient="records"),
    "contrasts": df_contrasts.to_dict(orient="records"),
    "per_class": df_per_class.to_dict(orient="records"),
    "transfer_matrix": df_transfer.to_dict(orient="records"),
    "old_vs_new": df_ovn.to_dict(orient="records"),
    "class_dropout": df_drop.to_dict(orient="records"),
}

with open(TABLES_DIR / "T1_canonical_results.json", "w", encoding="utf-8") as f:
    json.dump(canonical_data, f, indent=2)
print("Saved tables/T1_canonical_results.json")

# --- Step 8: Strict Acceptance Assertions ---
print("\n=== Verifying Acceptance Assertions for T1 ===")

# 1. 120 runs loaded
assert len(conf_new["RF"]["I28"]["t15"]) == 3, "Missing runs"

# 2. RF I28 cross_ba in [66.9, 67.2]
rf_i28_cba = df_main_new[(df_main_new.learner == "RF") & (df_main_new.representation == "I28")]["cross_ba_mean"].iloc[0]
print(f"Assertion 2: RF I28 cross_ba = {rf_i28_cba:.4f} (expected [66.9, 67.2])")
assert 66.9 <= rf_i28_cba <= 67.2, f"RF I28 cross_ba {rf_i28_cba} out of range [66.9, 67.2]"

# 3. RF I28_G29 cross_ba in [73.2, 73.5]
rf_g29_cba = df_main_new[(df_main_new.learner == "RF") & (df_main_new.representation == "I28_G29")]["cross_ba_mean"].iloc[0]
print(f"Assertion 3: RF I28_G29 cross_ba = {rf_g29_cba:.4f} (expected [73.2, 73.5])")
assert 73.2 <= rf_g29_cba <= 73.5, f"RF I28_G29 cross_ba {rf_g29_cba} out of range [73.2, 73.5]"

# 4. abs(RF C-main delta BA point estimate - 6.30) < 0.05
rf_cmain_dba = rf_g29_cba - rf_i28_cba
print(f"Assertion 4: RF C-main delta BA = {rf_cmain_dba:.4f} (expected diff < 0.05 from 6.30)")
assert abs(rf_cmain_dba - 6.30) < 0.05, f"RF delta BA {rf_cmain_dba} diff from 6.30 exceeds 0.05"

# 5. ci_lo < point_estimate < ci_hi for every contrast
for idx, r in df_contrasts.iterrows():
    pt = r["point_estimate"]
    lo = r["ci_lo"]
    hi = r["ci_hi"]
    print(f"  Contrast {r['contrast_id']} ({r['learner']} {r['metric']}): [{lo:.4f}, {hi:.4f}] contains pt {pt:.4f}: {lo <= pt <= hi}")
    assert lo <= pt <= hi, f"Contrast {r['contrast_id']} ({r['learner']} {r['metric']}) pt {pt} not in [{lo}, {hi}]"

print("=== ALL T1 ACCEPTANCE ASSERTIONS PASSED WITH ZERO ERROR ===")

# Log to T_PROGRESS.log
with open(WORKSPACE / "T_PROGRESS.log", "a", encoding="utf-8") as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] T1: Generated canonical results from runs_R1. RF dBA = +{rf_cmain_dba:.2f} pp, paired = +{df_ovn[(df_ovn.learner=='RF') & (df_ovn.metric=='BA')]['new_delta_cross_minus_delta_in'].iloc[0]:.2f} pp. All assertions passed.\n")
print("Logged to T_PROGRESS.log")
