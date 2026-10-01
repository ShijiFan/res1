"""
Phase P0-A: External Seasonal Validation on Autumn 2024 Datacube.
Protocol: configs_frozen/T5_freeze.json
Evaluates frozen Spring 2024 canonical models (RF, SVC across 3 seeds) on the Autumn 2024 datacube.

Outputs:
- runs_ext/<learner>_<rep>_<src>_s<seed>/predictions.parquet and blocks_confusion.npy
- tables/table_external_validation.csv
- results/robustness/external_autumn_evaluation_detailed.csv
- figures/fig_E_external_validation.pdf and .png
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, sobel
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Derived 2026-09-29 from tgrs_revision/scripts/run_external_autumn_eval.py for the v7 rerun.
# Changes: paths only. Models = v7 spring runs; cube = cube_4track_autumn_v7; all outputs under ws/autumn;
# the protocol freeze file is read from the original location; nothing is copied into a manuscript dir.
_ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = _ROOT.parent / "tgrs_revision"
RUNS_R1 = _ROOT / "ws" / "runs_R1"
SAR_V2 = _ROOT.parent / "sar_v2"
AUTUMN_DIR = _ROOT / "data"
_OUT = _ROOT / "ws" / "autumn"
RUNS_EXT = _OUT / "runs_ext"
TABLES_DIR = _OUT / "tables"
ROBUST_DIR = _OUT / "robustness"
FIG_DIR = _OUT / "figures"
MANU_FIG_DIR = _OUT / "_no_manuscript_copy"

RUNS_EXT.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
ROBUST_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TR = ["t15", "t37", "t88", "t139"]
REPS = ["I28", "I28_G29"]
SEEDS = [17, 29, 43]
CLASSES = ["Forest", "Grassland", "Cropland", "Urban", "Water"]

def compute_ba(cm_5x5):
    recalls = []
    for k in range(5):
        denom = cm_5x5[k].sum()
        recalls.append(cm_5x5[k, k] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls) * 100.0)

def extract_features_from_cube(c_autumn, vi):
    print("Extracting Autumn 2024 features matching canonical R1 definitions...")
    sensor_mask = c_autumn['four_track_valid']
    
    def L(k):
        return np.nan_to_num(c_autumn[k].astype(np.float64))
    
    def lstd_p(a, w=3):
        x = np.where(sensor_mask, np.nan_to_num(a), 0.0).astype(np.float64)
        m = sensor_mask.astype(np.float64)
        c = uniform_filter(m, w, mode='nearest')
        mu = np.divide(uniform_filter(x, w, mode='nearest'), c, out=np.zeros_like(c), where=c > 0)
        mu2 = np.divide(uniform_filter(x * x, w, mode='nearest'), c, out=np.zeros_like(c), where=c > 0)
        return np.sqrt(np.clip(mu2 - mu * mu, 0, None))

    def gradm_p(a):
        x = np.where(sensor_mask, np.nan_to_num(a), 0.0).astype(np.float32)
        full_valid = uniform_filter(sensor_mask.astype(np.float32), 3, mode='constant') > 0.99
        gx = sobel(x, axis=0, mode='nearest')
        gy = sobel(x, axis=1, mode='nearest')
        return np.where(full_valid, np.hypot(gx, gy), 0.0)

    track_feats = {r: {} for r in REPS}
    for t in TR:
        vv1, vh1, vv2, vh2 = [L(f"{t}_{k}") for k in ('d1_vv', 'd1_vh', 'd2_vv', 'd2_vh')]
        vvm = (vv1 + vv2) / 2.0
        vhm = (vh1 + vh2) / 2.0
        rtm = ((vh1 - vv1) + (vh2 - vv2)) / 2.0
        vvd = vv2 - vv1
        vhd = vh2 - vh1
        rtd = (vh2 - vv2) - (vh1 - vv1)
        g12 = np.clip(L(f"{t}_g12"), 1e-4, 0.999)
        
        r_base8 = [vvm, vhm, rtm, vvd, vhd, rtd, lstd_p(vvm, 3), lstd_p(vhm, 3)]
        
        r_full_int = r_base8 + [
            np.abs(vvd), np.abs(vhd), np.abs(rtd),
            np.minimum(vv1, vv2), np.maximum(vv1, vv2),
            np.minimum(vh1, vh2), np.maximum(vh1, vh2),
            lstd_p(vvm, 5), lstd_p(vhm, 5),
            lstd_p(vvm, 7), lstd_p(vhm, 7),
            gradm_p(vvm), gradm_p(vhm), gradm_p(rtm),
            lstd_p(vvd, 3), lstd_p(rtm, 3),
            vv1, vv2, vh1, vh2
        ]
        
        r_full_int_g29 = r_full_int + [g12]
        
        def pack(lyrs):
            return np.nan_to_num(np.column_stack([np.asarray(l)[vi] for l in lyrs]), nan=0.0, posinf=0.0, neginf=0.0)
            
        track_feats["I28"][t] = pack(r_full_int)
        track_feats["I28_G29"][t] = pack(r_full_int_g29)
        print(f"  Track {t.upper()}: I28 shape {track_feats['I28'][t].shape}, I28_G29 shape {track_feats['I28_G29'][t].shape}")
        
    return track_feats

def run_external_validation():
    print("=" * 78)
    print("Phase P0-A: External Seasonal Validation (Autumn 2024 Datacube)")
    print("=" * 78)
    
    autumn_cube_path = AUTUMN_DIR / "cube_4track_autumn_v7.npz"
    if not autumn_cube_path.exists():
        print(f"ERROR: {autumn_cube_path} does not exist yet. Please build the autumn cube first.")
        sys.exit(1)
        
    # Verify freeze config
    freeze_json = REVISION_DIR / "configs_frozen" / "T5_freeze.json"
    assert freeze_json.exists(), f"Missing {freeze_json}"
    with open(freeze_json) as f:
        freeze_meta = json.load(f)
    print(f"Loaded protocol: {freeze_meta['protocol_name']} (Frozen at: {freeze_meta['frozen_at_utc']})")
    
    # Load Autumn datacube
    c_aut = np.load(autumn_cube_path)
    
    # Canonical MMU and test indices from runs_R1
    mmu = np.load(SAR_V2 / "mmu" / "mmu_mask.npz")
    vi = np.where(mmu['mmu_valid'])
    
    canonical_pred_sample = pd.read_parquet(RUNS_R1 / "RF_I28_t15_s17" / "predictions.parquet")
    sub_eval = canonical_pred_sample[canonical_pred_sample["target_track"] == "t15"].sort_values("pixel_index").reset_index(drop=True)
    eval_pix_indices = sub_eval["pixel_index"].values
    eval_block_ids = sub_eval["block_id"].values
    y_eval = sub_eval["y_true"].values
    unique_blocks = sorted(list(set(eval_block_ids)))
    
    print(f"Canonical evaluation set: {len(eval_pix_indices)} pixels across {len(unique_blocks)} blocks: {unique_blocks}")
    
    # Check valid support on Autumn cube for these 9,600 pixels
    aut_valid_pixels = c_aut["four_track_valid"][vi][eval_pix_indices]
    valid_count = int(aut_valid_pixels.sum())
    valid_pct = 100.0 * valid_count / len(eval_pix_indices)
    print(f"Autumn valid support on test pixels: {valid_count} / {len(eval_pix_indices)} ({valid_pct:.2f}%)")
    
    if valid_pct < 100.0:
        print(f"Filtering evaluation set to {valid_count} fully valid pixels on Autumn cube.")
        use_mask = aut_valid_pixels
        eval_pix_indices_use = eval_pix_indices[use_mask]
        eval_block_ids_use = eval_block_ids[use_mask]
        y_eval_use = y_eval[use_mask]
    else:
        eval_pix_indices_use = eval_pix_indices
        eval_block_ids_use = eval_block_ids
        y_eval_use = y_eval
        
    print(f"Final evaluation pixels: N = {len(y_eval_use)}")
    for cl_id, cl_name in enumerate(CLASSES):
        n_cl = int((y_eval_use == cl_id).sum())
        print(f"  Class {cl_id} ({cl_name:10s}): {n_cl:5d} px ({100*n_cl/len(y_eval_use):.2f}%)")
        
    # Extract features from Autumn cube
    track_feats_aut = extract_features_from_cube(c_aut, vi)
    
    # Evaluate frozen models
    detailed_records = []
    total_evals = len(["RF", "SVC"]) * len(SEEDS) * len(REPS) * len(TR) * len(TR)
    print(f"\nEvaluating {total_evals} model-pair combinations...")
    
    for learner in ["RF", "SVC"]:
        for rep in REPS:
            for src in TR:
                for seed in SEEDS:
                    model_dir = RUNS_R1 / f"{learner}_{rep}_{src}_s{seed}"
                    m_path = model_dir / "model.joblib"
                    scaler_path = model_dir / "scaler.joblib"
                    
                    if not m_path.exists():
                        print(f"FATAL: Missing model {m_path}")
                        sys.exit(1)
                        
                    model = joblib.load(m_path)
                    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
                    
                    # Prepare export directory for runs_ext
                    ext_dir = RUNS_EXT / f"{learner}_{rep}_{src}_s{seed}"
                    ext_dir.mkdir(parents=True, exist_ok=True)
                    
                    parquet_dfs = []
                    blocks_confusion = np.zeros((len(unique_blocks), 5, 5), dtype=np.int32)
                    blocks_oa = np.zeros(len(unique_blocks), dtype=np.float64)
                    
                    for tgt in TR:
                        X_aut = track_feats_aut[rep][tgt][eval_pix_indices_use]
                        if scaler is not None:
                            X_aut = scaler.transform(X_aut)
                            
                        preds = model.predict(X_aut)
                        cm = confusion_matrix(y_eval_use, preds, labels=[0,1,2,3,4])
                        oa = accuracy_score(y_eval_use, preds) * 100.0
                        ba = compute_ba(cm)
                        
                        rec = {
                            "learner": learner,
                            "seed": seed,
                            "representation": rep,
                            "source_track": src,
                            "target_track": tgt,
                            "is_cross": bool(src != tgt),
                            "n_eval": len(y_eval_use),
                            "oa": oa,
                            "ba": ba,
                            "rec_forest": float(cm[0, 0] / cm[0].sum() * 100.0) if cm[0].sum() > 0 else 0.0,
                            "rec_grassland": float(cm[1, 1] / cm[1].sum() * 100.0) if cm[1].sum() > 0 else 0.0,
                            "rec_cropland": float(cm[2, 2] / cm[2].sum() * 100.0) if cm[2].sum() > 0 else 0.0,
                            "rec_urban": float(cm[3, 3] / cm[3].sum() * 100.0) if cm[3].sum() > 0 else 0.0,
                            "rec_water": float(cm[4, 4] / cm[4].sum() * 100.0) if cm[4].sum() > 0 else 0.0,
                        }
                        detailed_records.append(rec)
                        
                        df_pred_tgt = pd.DataFrame({
                            "target_track": tgt,
                            "block_id": eval_block_ids_use,
                            "pixel_index": eval_pix_indices_use,
                            "y_true": y_eval_use,
                            "y_pred": preds
                        })
                        parquet_dfs.append(df_pred_tgt)
                        
                    # Save per-run artifacts to runs_ext
                    all_pred_df = pd.concat(parquet_dfs, ignore_index=True)
                    all_pred_df.to_parquet(ext_dir / "predictions.parquet", index=False)
                    
                    # Compute per-block confusions across target tracks for record
                    for b_idx, b_val in enumerate(unique_blocks):
                        b_mask = (all_pred_df["block_id"] == b_val)
                        b_true = all_pred_df.loc[b_mask, "y_true"]
                        b_pred = all_pred_df.loc[b_mask, "y_pred"]
                        blocks_confusion[b_idx] = confusion_matrix(b_true, b_pred, labels=[0,1,2,3,4])
                        blocks_oa[b_idx] = accuracy_score(b_true, b_pred) * 100.0
                        
                    np.save(ext_dir / "blocks_confusion.npy", blocks_confusion)
                    np.save(ext_dir / "blocks_oa.npy", blocks_oa)
                    
    df_det = pd.DataFrame(detailed_records)
    det_csv = ROBUST_DIR / "external_autumn_evaluation_detailed.csv"
    df_det.to_csv(det_csv, index=False)
    print(f"\nSaved detailed evaluation records to {det_csv}")
    
    # -------------------------------------------------------------
    # Summaries and Contrasts
    # -------------------------------------------------------------
    df_cross = df_det[df_det["is_cross"]].copy()
    df_in = df_det[~df_det["is_cross"]].copy()
    
    print("\n" + "=" * 78)
    print("AUTUMN 2024 EXTERNAL VALIDATION RESULTS (CROSS-TRACK)")
    print("=" * 78)
    
    contrast_rows = []
    
    for learner in ["RF", "SVC"]:
        sub_c = df_cross[df_cross["learner"] == learner]
        sub_in = df_in[df_in["learner"] == learner]
        
        piv_c = sub_c.pivot(index=["seed", "source_track", "target_track"], columns="representation")
        piv_in = sub_in.pivot(index=["seed", "source_track", "target_track"], columns="representation")
        
        d_ba_c = piv_c["ba"]["I28_G29"] - piv_c["ba"]["I28"]
        d_oa_c = piv_c["oa"]["I28_G29"] - piv_c["oa"]["I28"]
        d_urb_c = piv_c["rec_urban"]["I28_G29"] - piv_c["rec_urban"]["I28"]
        d_for_c = piv_c["rec_forest"]["I28_G29"] - piv_c["rec_forest"]["I28"]
        
        d_ba_in = piv_in["ba"]["I28_G29"] - piv_in["ba"]["I28"]
        delta_transfer = d_ba_c.mean() - d_ba_in.mean()
        
        ci_ba = [np.percentile(d_ba_c, 2.5), np.percentile(d_ba_c, 97.5)]
        ci_oa = [np.percentile(d_oa_c, 2.5), np.percentile(d_oa_c, 97.5)]
        ci_trans = [np.percentile(d_ba_c.values - d_ba_in.mean(), 2.5), np.percentile(d_ba_c.values - d_ba_in.mean(), 97.5)]
        
        rec_row = {
            "learner": learner,
            "season": "Autumn 2024",
            "i28_cross_ba": round(float(piv_c["ba"]["I28"].mean()), 2),
            "i28_g12_cross_ba": round(float(piv_c["ba"]["I28_G29"].mean()), 2),
            "delta_ba": round(float(d_ba_c.mean()), 2),
            "ci_95_delta_ba": f"[{ci_ba[0]:.2f}, {ci_ba[1]:.2f}]",
            "i28_cross_oa": round(float(piv_c["oa"]["I28"].mean()), 2),
            "i28_g12_cross_oa": round(float(piv_c["oa"]["I28_G29"].mean()), 2),
            "delta_oa": round(float(d_oa_c.mean()), 2),
            "ci_95_delta_oa": f"[{ci_oa[0]:.2f}, {ci_oa[1]:.2f}]",
            "delta_urban_recall": round(float(d_urb_c.mean()), 2),
            "delta_forest_recall": round(float(d_for_c.mean()), 2),
            "transfer_stabilization": round(float(delta_transfer), 2),
            "ci_95_transfer": f"[{ci_trans[0]:.2f}, {ci_trans[1]:.2f}]",
            "excludes_zero": bool(ci_ba[0] > 0),
            "verdict": "CONFIRMATION HOLDS" if ci_ba[0] > 0 else ("DIRECTIONAL CONSISTENT" if d_ba_c.mean() > 0 else "FAILS")
        }
        contrast_rows.append(rec_row)
        
    df_table = pd.DataFrame(contrast_rows)
    table_csv = TABLES_DIR / "table_external_validation.csv"
    df_table.to_csv(table_csv, index=False)
    print(df_table.to_string())
    print(f"\nSaved summary table to {table_csv}")
    
    # -------------------------------------------------------------
    # Plot Figure E: External Seasonal Validation
    # -------------------------------------------------------------
    print("\nGenerating Figure E: External Seasonal Validation...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    colors = {"RF": "#1f77b4", "SVC": "#2ca02c"}
    
    for idx, learner in enumerate(["RF", "SVC"]):
        ax = axes[idx]
        sub_c = df_cross[df_cross["learner"] == learner]
        piv_c = sub_c.pivot(index=["seed", "source_track", "target_track"], columns="representation")
        d_ba_c = piv_c["ba"]["I28_G29"] - piv_c["ba"]["I28"]
        
        # Histogram of Delta BA across the 36 (3 seeds x 12 track pairs)
        n, bins, patches = ax.hist(d_ba_c, bins=12, color=colors[learner], edgecolor="black", alpha=0.75,
                                   label=f"{learner} Distributions (N=36)")
        
        mean_val = d_ba_c.mean()
        ci_lo, ci_hi = np.percentile(d_ba_c, 2.5), np.percentile(d_ba_c, 97.5)
        
        ax.axvline(0, color="#d62728", linestyle=":", linewidth=2.0, label="Zero increment ($y=0$)")
        ax.axvline(mean_val, color="#000000", linestyle="--", linewidth=2.2,
                   label=f"Mean $\\Delta$BA: +{mean_val:.2f} pp")
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=colors[learner],
                   label=f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] pp")
        
        ax.set_title(f"({chr(97+idx)}) {learner}: Seasonal Transfer Increment $\\Delta\\text{{BA}}$", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\Delta\text{BA} = (I_{28}+g_{12}) - I_{28}$ (pp)", fontsize=10)
        ax.set_ylabel("Frequency (Source-Target Pairs $\\times$ Seeds)", fontsize=10)
        ax.legend(frameon=True, fontsize=8.5, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        
    plt.tight_layout()
    fig_e_png = FIG_DIR / "fig_E_external_validation.png"
    fig_e_pdf = FIG_DIR / "fig_E_external_validation.pdf"
    plt.savefig(fig_e_png, bbox_inches="tight", dpi=300)
    plt.savefig(fig_e_pdf, bbox_inches="tight")
    plt.close()
    
    if MANU_FIG_DIR.exists():
        import shutil
        shutil.copy(fig_e_png, MANU_FIG_DIR / fig_e_png.name)
        shutil.copy(fig_e_pdf, MANU_FIG_DIR / fig_e_pdf.name)
        
    print(f"Generated Figure E successfully: {fig_e_png} & {fig_e_pdf}")
    print("=" * 78)
    print("PHASE P0-A EXTERNAL SEASONAL VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 78)

if __name__ == "__main__":
    run_external_validation()
