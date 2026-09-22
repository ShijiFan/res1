"""Core Model Fitting & Full Artifact Preservation (Step 1B).
Fits 24 models:
  Learners: RF, SVC
  Representations: Full_Intensity_28, Base8_Raw_g12, F2_D12
  Sources: t15, t37, t88, t139
  Seed: 17
Persists all models, scalers, parquet predictions, and confusion arrays to disk.
"""
import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.ndimage import uniform_filter, sobel

ROOT = Path(r"E:\research\SAR")
EXP = ROOT / "revision_experiments_20260914"
SAR_V2 = ROOT / "sar_v2"
WORK_DIR = ROOT / "tgrs_evidence_repair_20260915"
RUNS_DIR = WORK_DIR / "runs"
CONFIGS_DIR = WORK_DIR / "configs"
TABLES_DIR = WORK_DIR / "tables"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TR = ['t15', 't37', 't88', 't139']
REPS = ['Full_Intensity_28', 'Base8_Raw_g12', 'F2_D12']
LEARNERS = ['RF', 'SVC']
SEED = 17

def main():
    start_time_iso = datetime.now(timezone.utc).isoformat()
    print("===================================================================")
    print("=== TGRS EVIDENCE REPAIR: FITTING 24 CORE MODELS (SEED 17) ===")
    print("===================================================================")
    
    # 1. Load Data
    print("Loading datacube and metadata...")
    C6 = np.load(SAR_V2 / "data" / "cube_4track_v6.npz")
    v4 = np.load(SAR_V2 / "data" / "flevoland_datacube_v4_multibaseline.npz")
    mv = np.load(SAR_V2 / "mmu" / "mmu_mask.npz")['mmu_valid']
    vi = np.where(mv)
    OM = np.load(EXP / "omega_v6_rebuilt.npy")
    omv = OM[vi]
    cm_labels = np.load(SAR_V2 / "labels" / "consensus_mask.npz")
    
    # Global grid coordinates
    grid_y = v4['grid_y'][vi]
    grid_x = v4['grid_x'][vi]
    bids = ((grid_y // 125) * 100 + grid_x // 125)
    
    # Labels (0..4)
    y_wc = v4['y'][vi].astype(int)
    y_clc = (cm_labels['clc_on_mmu'] - 1).astype(int)
    
    # Consensus mask
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    y_true_all = y_wc
    
    # Evaluation blocks
    with open(EXP / "splits" / "historical_eval_blocks.json", "r") as f:
        split_info = json.load(f)
    trb = sorted(split_info["train_blocks"])
    teb = sorted(split_info["eval_blocks"])
    NB = len(teb)
    
    print(f"Total consensus pixels: {c_mask.sum():,}")
    print(f"Train blocks: {len(trb)}, Eval blocks: {NB}")
    
    # 2. Extract Features using Pure Sensor Mask
    sensor_mask = C6['four_track_valid']
    
    def L(k):
        return np.nan_to_num(C6[k].astype(np.float64))
    
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

    print("Extracting feature layers for all 4 tracks...")
    track_feats = {r: {} for r in REPS}
    feature_names = {}
    
    for t in TR:
        vv1, vh1, vv2, vh2 = [L(f"{t}_{k}") for k in ('d1_vv', 'd1_vh', 'd2_vv', 'd2_vh')]
        vvm = (vv1 + vv2) / 2.0
        vhm = (vh1 + vh2) / 2.0
        rtm = ((vh1 - vv1) + (vh2 - vv2)) / 2.0
        vvd = vv2 - vv1
        vhd = vh2 - vh1
        rtd = (vh2 - vv2) - (vh1 - vv1)
        g12 = np.clip(L(f"{t}_g12"), 1e-4, 0.999)
        
        # Base 8
        r_base8 = [vvm, vhm, rtm, vvd, vhd, rtd, lstd_p(vvm, 3), lstd_p(vhm, 3)]
        base8_names = ["vvm", "vhm", "rtm", "vvd", "vhd", "rtd", "lstd_vvm_3", "lstd_vhm_3"]
        
        # Full Intensity 28
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
        full_int_names = base8_names + [
            "abs_vvd", "abs_vhd", "abs_rtd",
            "min_vv", "max_vv", "min_vh", "max_vh",
            "lstd_vvm_5", "lstd_vhm_5", "lstd_vvm_7", "lstd_vhm_7",
            "gradm_vvm", "gradm_vhm", "gradm_rtm",
            "lstd_vvd_3", "lstd_rtm_3",
            "vv1", "vv2", "vh1", "vh2"
        ]
        
        # Base8 + Raw g12 (9-D)
        r_base8_g12 = r_base8 + [g12]
        base8_g12_names = base8_names + ["g12_raw"]
        
        # F2_D12 (12-D)
        r_f2_d12 = r_base8 + [
            g12,
            lstd_p(g12, 3),
            gradm_p(g12),
            np.log(1.0 - g12 + 1e-4)
        ]
        f2_d12_names = base8_names + ["g12", "lstd_g12_3", "gradm_g12", "log_neg_g12"]
        
        def pack(lyrs):
            return np.nan_to_num(np.column_stack([np.asarray(l)[vi] for l in lyrs]), nan=0.0, posinf=0.0, neginf=0.0)
        
        track_feats['Full_Intensity_28'][t] = pack(r_full_int)
        track_feats['Base8_Raw_g12'][t] = pack(r_base8_g12)
        track_feats['F2_D12'][t] = pack(r_f2_d12)
        
        feature_names['Full_Intensity_28'] = full_int_names
        feature_names['Base8_Raw_g12'] = base8_g12_names
        feature_names['F2_D12'] = f2_d12_names

    print(f"Features ready. Dimensions: Full_Intensity_28={len(feature_names['Full_Intensity_28'])}, Base8_Raw_g12={len(feature_names['Base8_Raw_g12'])}, F2_D12={len(feature_names['F2_D12'])}")
    
    # 3. Training and Evaluation Sample Sets
    # Training: 20,000 pixels sampled from trb in consensus C
    tr_cand = np.where(np.isin(bids, trb) & c_mask)[0]
    i_tr = np.random.RandomState(SEED).choice(tr_cand, 20000, replace=False)
    y_train = y_true_all[i_tr]
    
    # Compute SHA-256 hash of training pixel indices
    tr_hash = hashlib.sha256(i_tr.tobytes()).hexdigest()
    print(f"Training samples: {len(i_tr)}, pixel hash: {tr_hash[:16]}...")
    
    # Evaluation: 800 pixels per block in teb
    eval_pixels = {}
    eval_all_indices = []
    for b in teb:
        ix = np.where((bids == b) & c_mask)[0]
        if len(ix) > 800:
            eval_pixels[b] = np.random.RandomState(42).choice(ix, 800, replace=False)
        else:
            eval_pixels[b] = ix
        eval_all_indices.extend(eval_pixels[b])
    
    print(f"Evaluation blocks: {NB}, Total unique test pixels: {len(eval_all_indices):,}")

    # 4. Training Loop: 24 Models
    run_records = []
    total_models = len(LEARNERS) * len(REPS) * len(TR)
    model_count = 0
    
    for lrn in LEARNERS:
        for rep in REPS:
            for src in TR:
                model_count += 1
                setting_key = f"{lrn}_{rep}_{src}_s{SEED}"
                setting_dir = RUNS_DIR / setting_key
                setting_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"[{model_count:02d}/{total_models}] Fitting: {setting_key} ...")
                t_fit_start = time.time()
                
                X_src_tr = track_feats[rep][src][i_tr]
                
                # Scaler & Classifier Setup
                if lrn == "SVC":
                    scaler = RobustScaler().fit(X_src_tr)
                    X_tr_proc = scaler.transform(X_src_tr)
                    clf = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
                else: # RF
                    scaler = None
                    X_tr_proc = X_src_tr
                    clf = RandomForestClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42
                    )
                
                # Train model
                clf.fit(X_tr_proc, y_train)
                fit_duration = time.time() - t_fit_start
                
                # Persist model and scaler
                joblib.dump(clf, setting_dir / "model.joblib")
                if scaler is not None:
                    joblib.dump(scaler, setting_dir / "scaler.joblib")
                
                # Predict on all 4 target tracks across 12 blocks
                blocks_confusion = np.zeros((4, NB, 5, 5), dtype=np.int32)
                blocks_oa = np.zeros((4, NB), dtype=np.float64)
                
                pred_rows = []
                for ti, tgt in enumerate(TR):
                    X_tgt = track_feats[rep][tgt]
                    for bi_, b in enumerate(teb):
                        ix_eval = eval_pixels[b]
                        if len(ix_eval) == 0:
                            continue
                        
                        x_ev = X_tgt[ix_eval]
                        if scaler is not None:
                            x_ev = scaler.transform(x_ev)
                        y_pred = clf.predict(x_ev)
                        y_true_b = y_true_all[ix_eval]
                        
                        # Confusion & OA
                        cm_b = confusion_matrix(y_true_b, y_pred, labels=[0, 1, 2, 3, 4])
                        blocks_confusion[ti, bi_] = cm_b
                        blocks_oa[ti, bi_] = accuracy_score(y_true_b, y_pred)
                        
                        # Parquet records
                        r_coords = grid_y[ix_eval]
                        c_coords = grid_x[ix_eval]
                        for k in range(len(ix_eval)):
                            pred_rows.append({
                                "region": "Site-1_Flevoland",
                                "window": "Window-1_May2024",
                                "source": src,
                                "target": tgt,
                                "block_id": int(b),
                                "pixel_id": int(ix_eval[k]),
                                "global_row": int(r_coords[k]),
                                "global_col": int(c_coords[k]),
                                "truth": int(y_true_b[k]),
                                "pred": int(y_pred[k])
                            })
                
                # Save numpy confusion & OA arrays
                np.save(setting_dir / "blocks_confusion.npy", blocks_confusion)
                np.save(setting_dir / "blocks_oa.npy", blocks_oa)
                
                # Save Parquet predictions
                df_pred = pd.DataFrame(pred_rows)
                pq_table = pa.Table.from_pandas(df_pred)
                pq.write_table(pq_table, setting_dir / "predictions.parquet", compression="snappy")
                
                # Save config & run metadata
                run_config = {
                    "setting_key": setting_key,
                    "learner": lrn,
                    "representation": rep,
                    "source_track": src,
                    "seed": SEED,
                    "n_training_samples": len(i_tr),
                    "training_pixel_hash": tr_hash,
                    "feature_names": feature_names[rep],
                    "n_features": len(feature_names[rep]),
                    "hyperparameters": clf.get_params(),
                    "eval_blocks": teb,
                    "total_predictions": len(df_pred),
                    "fit_duration_seconds": float(fit_duration),
                    "python_version": sys.version,
                    "sklearn_version": "1.7.2",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": 0
                }
                with open(setting_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump(run_config, f, indent=2)
                
                # Compute quick summary metrics for verification
                # Cross-track OA (target != src)
                cross_targets = [ti for ti, tgt in enumerate(TR) if tgt != src]
                cross_oa = float(np.mean(blocks_oa[cross_targets])) * 100.0
                intra_target = TR.index(src)
                intra_oa = float(np.mean(blocks_oa[intra_target])) * 100.0
                
                run_records.append({
                    "setting_key": setting_key,
                    "learner": lrn,
                    "representation": rep,
                    "source": src,
                    "cross_oa": cross_oa,
                    "intra_oa": intra_oa,
                    "fit_duration_sec": fit_duration,
                    "predictions_count": len(df_pred)
                })
                print(f"       -> Cross OA: {cross_oa:.2f}% | Intra OA: {intra_oa:.2f}% | Parquet rows: {len(df_pred):,} | Time: {fit_duration:.1f}s")

    print("\nAll 24 core models trained and fully persisted to disk.")
    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(TABLES_DIR / "runs_inventory.csv", index=False)
    print("Saved runs_inventory.csv")

if __name__ == "__main__":
    main()
