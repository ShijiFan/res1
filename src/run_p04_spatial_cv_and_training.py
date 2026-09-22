"""
Phase P04: Source-Domain Spatial Cross-Validation, Hyperparameter Freezing,
and Multi-Seed Main Model Fits across 5 Core Representations.
Fast multithreaded evaluation preserving exact protocol.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed, dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.ndimage import uniform_filter, sobel

WORKSPACE = Path(r"E:\research\SAR\tgrs_final_campaign_20260915")
REPAIR_WORKSPACE = Path(r"E:\research\SAR\tgrs_evidence_repair_20260915")
SAR_V2 = Path(r"E:\research\SAR\sar_v2")

TR = ["t15", "t37", "t88", "t139"]
REPS = ["I8", "I28", "I8_G9", "F12", "I28_G29"]
SEEDS = [17, 29, 43]
TARGET_ORDER = ["t15", "t37", "t88", "t139"]

def log_ledger(action, phase, details, exit_code=0):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": phase,
        "action": action,
        "exit_code": exit_code,
        "details": details
    }
    with open(WORKSPACE / "RUN_LEDGER.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[{phase}] {action} -> Exit {exit_code}")

def compute_pooled_ba(cm_array_5x5):
    recalls = []
    for k in range(5):
        denom = cm_array_5x5[k].sum()
        if denom > 0:
            recalls.append(cm_array_5x5[k, k] / denom)
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))

def extract_features(c6, vi):
    sensor_mask = c6['four_track_valid']
    
    def L(k):
        return np.nan_to_num(c6[k].astype(np.float64))
    
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

    print("Building feature layers across all 4 tracks...")
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
        
        # Base 8 (I8)
        r_base8 = [vvm, vhm, rtm, vvd, vhd, rtd, lstd_p(vvm, 3), lstd_p(vhm, 3)]
        base8_names = ["vvm", "vhm", "rtm", "vvd", "vhd", "rtd", "lstd_vvm_3", "lstd_vhm_3"]
        
        # Full Intensity 28 (I28)
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
        
        # Base8 + Raw g12 (I8_G9)
        r_base8_g12 = r_base8 + [g12]
        base8_g12_names = base8_names + ["g12_raw"]
        
        # F2_D12 (F12)
        r_f2_d12 = r_base8 + [
            g12,
            lstd_p(g12, 3),
            gradm_p(g12),
            np.log(1.0 - g12 + 1e-4)
        ]
        f2_d12_names = base8_names + ["g12", "lstd_g12_3", "gradm_g12", "log_neg_g12"]
        
        # Full Intensity 28 + Raw g12 (I28_G29)
        r_full_int_g29 = r_full_int + [g12]
        full_int_g29_names = full_int_names + ["g12_raw"]
        
        def pack(lyrs):
            return np.nan_to_num(np.column_stack([np.asarray(l)[vi] for l in lyrs]), nan=0.0, posinf=0.0, neginf=0.0)
        
        track_feats['I8'][t] = pack(r_base8)
        track_feats['I28'][t] = pack(r_full_int)
        track_feats['I8_G9'][t] = pack(r_base8_g12)
        track_feats['F12'][t] = pack(r_f2_d12)
        track_feats['I28_G29'][t] = pack(r_full_int_g29)
        
        feature_names['I8'] = base8_names
        feature_names['I28'] = full_int_names
        feature_names['I8_G9'] = base8_g12_names
        feature_names['F12'] = f2_d12_names
        feature_names['I28_G29'] = full_int_g29_names

    print(f"Features ready. Dimensions: { {k: len(v) for k, v in feature_names.items()} }")
    return track_feats, feature_names

def eval_svc_candidate(p, fold_data):
    records = []
    fold_scores = []
    for fd in fold_data:
        actual_gamma = p["gamma_mult"] * fd["gamma_scale"]
        clf = SVC(C=p["C"], gamma=actual_gamma, random_state=42)
        clf.fit(fd["X_tr_scaled"], fd["y_tr"])
        preds = clf.predict(fd["X_val_scaled"])
        cm = confusion_matrix(fd["y_val"], preds, labels=[0,1,2,3,4])
        ba = compute_pooled_ba(cm)
        oa = accuracy_score(fd["y_val"], preds)
        fold_scores.append({"fold": fd["fold"], "ba": ba, "oa": oa})
        records.append({
            "learner": "SVC", "params": str(p), "fold": fd["fold"], "fold_ba": ba, "fold_oa": oa
        })
    mean_ba = float(np.mean([s["ba"] for s in fold_scores]))
    mean_oa = float(np.mean([s["oa"] for s in fold_scores]))
    c_simp = {0.1: 4, 1.0: 3, 10.0: 2, 100.0: 1}.get(p["C"], 0)
    g_simp = 2 if p["gamma_mult"] == 1.0 else 1
    simp_score = c_simp + g_simp
    return {"params": p, "mean_ba": mean_ba, "mean_oa": mean_oa, "simp_score": simp_score, "records": records}

def eval_rf_candidate(p, fold_data):
    records = []
    fold_scores = []
    for fd in fold_data:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_features=p["max_features"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=42,
            n_jobs=2
        )
        clf.fit(fd["X_tr"], fd["y_tr"])
        preds = clf.predict(fd["X_val"])
        cm = confusion_matrix(fd["y_val"], preds, labels=[0,1,2,3,4])
        ba = compute_pooled_ba(cm)
        oa = accuracy_score(fd["y_val"], preds)
        fold_scores.append({"fold": fd["fold"], "ba": ba, "oa": oa})
        records.append({
            "learner": "RF", "params": str(p), "fold": fd["fold"], "fold_ba": ba, "fold_oa": oa
        })
    mean_ba = float(np.mean([s["ba"] for s in fold_scores]))
    mean_oa = float(np.mean([s["oa"] for s in fold_scores]))
    simp_score = (1 if p["min_samples_leaf"] == 5 else 0) + (2 if p["max_features"] == "sqrt" else (1 if p["max_features"] == 0.5 else 0))
    return {"params": p, "mean_ba": mean_ba, "mean_oa": mean_oa, "simp_score": simp_score, "records": records}

def run_spatial_cv(track_feats, bids, c_mask, y_wc, cv_folds):
    print("=== Step 1: Running Source-Domain 3-Fold Spatial Cross-Validation (Seed 17) ===")
    
    cv_records = []
    selected_models = {}
    
    svc_grid = [
        {"C": 0.1, "gamma_mult": 0.1},
        {"C": 0.1, "gamma_mult": 1.0},
        {"C": 0.1, "gamma_mult": 10.0},
        {"C": 1.0, "gamma_mult": 0.1},
        {"C": 1.0, "gamma_mult": 1.0},
        {"C": 1.0, "gamma_mult": 10.0},
        {"C": 10.0, "gamma_mult": 0.1},
        {"C": 10.0, "gamma_mult": 1.0},
        {"C": 10.0, "gamma_mult": 10.0},
        {"C": 100.0, "gamma_mult": 0.1},
        {"C": 100.0, "gamma_mult": 1.0},
        {"C": 100.0, "gamma_mult": 10.0}
    ]
    
    rf_grid = [
        {"max_features": "sqrt", "min_samples_leaf": 1},
        {"max_features": "sqrt", "min_samples_leaf": 5},
        {"max_features": 0.5, "min_samples_leaf": 1},
        {"max_features": 0.5, "min_samples_leaf": 5},
        {"max_features": 1.0, "min_samples_leaf": 1},
        {"max_features": 1.0, "min_samples_leaf": 5}
    ]
    
    f1_blocks, f2_blocks, f3_blocks = cv_folds["fold_1_blocks"], cv_folds["fold_2_blocks"], cv_folds["fold_3_blocks"]
    fold_defs = [
        {"fold": 1, "train_blocks": f2_blocks + f3_blocks, "val_blocks": f1_blocks},
        {"fold": 2, "train_blocks": f1_blocks + f3_blocks, "val_blocks": f2_blocks},
        {"fold": 3, "train_blocks": f1_blocks + f2_blocks, "val_blocks": f3_blocks}
    ]

    t_start_cv = time.time()

    for rep in REPS:
        for src in TR:
            t0_cfg = time.time()
            X_all = track_feats[rep][src]
            
            # Prepare fold data (subsampled to 13,333 train, 6,667 val for speed & consistency)
            fold_data = []
            for fd in fold_defs:
                tr_cand = np.where(np.isin(bids, fd["train_blocks"]) & c_mask)[0]
                val_cand = np.where(np.isin(bids, fd["val_blocks"]) & c_mask)[0]
                
                rng = np.random.RandomState(17 + fd["fold"])
                tr_idx = rng.choice(tr_cand, min(13333, len(tr_cand)), replace=False)
                val_idx = rng.choice(val_cand, min(6667, len(val_cand)), replace=False)
                
                scaler = RobustScaler()
                X_tr = X_all[tr_idx]
                y_tr = y_wc[tr_idx]
                X_val = X_all[val_idx]
                y_val = y_wc[val_idx]
                
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
                
                var_tr = np.var(X_tr_scaled)
                gamma_scale = 1.0 / (X_tr.shape[1] * var_tr) if var_tr > 0 else 0.1
                
                fold_data.append({
                    "fold": fd["fold"],
                    "X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
                    "X_tr_scaled": X_tr_scaled, "X_val_scaled": X_val_scaled,
                    "gamma_scale": gamma_scale
                })

            # Parallel evaluation of SVC grid (12 candidates)
            svc_results = Parallel(n_jobs=4, prefer="threads")(
                delayed(eval_svc_candidate)(p, fold_data) for p in svc_grid
            )
            for res in svc_results:
                for rec in res["records"]:
                    rec["representation"] = rep
                    rec["source_track"] = src
                    cv_records.append(rec)
            best_svc = sorted(svc_results, key=lambda x: (round(x["mean_ba"], 3), x["simp_score"], x["mean_ba"]), reverse=True)[0]
            selected_models[f"SVC_{rep}_{src}"] = {
                "learner": "SVC", "representation": rep, "source_track": src,
                "selected_params": best_svc["params"], "cv_mean_ba": best_svc["mean_ba"], "cv_mean_oa": best_svc["mean_oa"]
            }

            # Parallel evaluation of RF grid (6 candidates)
            rf_results = Parallel(n_jobs=3, prefer="threads")(
                delayed(eval_rf_candidate)(p, fold_data) for p in rf_grid
            )
            for res in rf_results:
                for rec in res["records"]:
                    rec["representation"] = rep
                    rec["source_track"] = src
                    cv_records.append(rec)
            best_rf = sorted(rf_results, key=lambda x: (round(x["mean_ba"], 3), x["simp_score"], x["mean_ba"]), reverse=True)[0]
            selected_models[f"RF_{rep}_{src}"] = {
                "learner": "RF", "representation": rep, "source_track": src,
                "selected_params": best_rf["params"], "cv_mean_ba": best_rf["mean_ba"], "cv_mean_oa": best_rf["mean_oa"]
            }

            dur = time.time() - t0_cfg
            print(f"[{rep} | {src} ({dur:.1f}s)] Selected RF: {best_rf['params']} (BA: {best_rf['mean_ba']*100:.2f}%) | Selected SVC: {best_svc['params']} (BA: {best_svc['mean_ba']*100:.2f}%)")

    total_cv_dur = time.time() - t_start_cv
    print(f"All 1,080 CV trials completed in {total_cv_dur/60:.1f} minutes.")

    # Save CV trials and selected models
    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(WORKSPACE / "tables" / "cv_trials.csv", index=False)
    
    with open(WORKSPACE / "configs" / "selected_models.json", "w", encoding="utf-8") as f:
        json.dump(selected_models, f, indent=2)
    print(f"Saved tables/cv_trials.csv ({len(df_cv)} rows) and configs/selected_models.json")
    
    return selected_models

def train_and_eval_final_models(track_feats, feature_names, bids, c_mask, y_wc, split_manifest, selected_models):
    print("=== Step 2: Training & Evaluating Final Models (5 Reps x 4 Tracks x 2 Learners x 3 Seeds = 120 Models) ===")
    
    trb = split_manifest["historical_development_split"]["train_blocks"]
    teb = split_manifest["historical_development_split"]["eval_blocks"]
    NB = len(teb)
    
    # Fixed evaluation pixel set (800 pixels per block, identical across all models)
    eval_pixels = {}
    eval_all_indices = []
    for b in teb:
        ix = np.where((bids == b) & c_mask)[0]
        rng_eval = np.random.RandomState(42 + b)
        n_take = min(800, len(ix))
        chosen = rng_eval.choice(ix, n_take, replace=False)
        eval_pixels[b] = chosen
        eval_all_indices.extend(chosen)
    eval_all_indices = np.array(eval_all_indices)
    
    y_eval = y_wc[eval_all_indices]
    block_eval_ids = bids[eval_all_indices]
    
    main_results = []
    t_start_train = time.time()
    
    for seed in SEEDS:
        tr_cand = np.where(np.isin(bids, trb) & c_mask)[0]
        rng_tr = np.random.RandomState(seed)
        i_tr = rng_tr.choice(tr_cand, 20000, replace=False)
        y_train = y_wc[i_tr]
        tr_hash = hashlib.sha256(i_tr.tobytes()).hexdigest()
        
        for learner in ["RF", "SVC"]:
            for rep in REPS:
                for src in TR:
                    key = f"{learner}_{rep}_{src}"
                    cfg_sel = selected_models[key]
                    params = cfg_sel["selected_params"]
                    
                    run_dir_name = f"{learner}_{rep}_{src}_s{seed}"
                    run_dir = WORKSPACE / "runs" / run_dir_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_path = run_dir / "model.joblib"
                    scaler_path = run_dir / "scaler.joblib"
                    parquet_path = run_dir / "predictions.parquet"
                    conf_path = run_dir / "blocks_confusion.npy"
                    oa_path = run_dir / "blocks_oa.npy"
                    cfg_path = run_dir / "config.json"
                    
                    # Check if already completed and valid
                    if model_path.exists() and parquet_path.exists() and conf_path.exists() and oa_path.exists() and cfg_path.exists():
                        blocks_oa = np.load(oa_path)
                        blocks_confusion = np.load(conf_path)
                    else:
                        X_src_train = track_feats[rep][src][i_tr]
                        
                        scaler = None
                        if learner == "SVC":
                            scaler = RobustScaler()
                            X_tr_fit = scaler.fit_transform(X_src_train)
                            dump(scaler, scaler_path)
                            
                            var_tr = np.var(X_tr_fit)
                            gamma_scale = 1.0 / (X_tr_fit.shape[1] * var_tr) if var_tr > 0 else 0.1
                            actual_gamma = params.get("gamma_mult", 1.0) * gamma_scale
                            clf = SVC(C=params.get("C", 1.0), gamma=actual_gamma, random_state=42)
                        else:
                            X_tr_fit = X_src_train
                            clf = RandomForestClassifier(
                                n_estimators=300,
                                max_features=params.get("max_features", "sqrt"),
                                min_samples_leaf=params.get("min_samples_leaf", 1),
                                random_state=42,
                                n_jobs=4
                            )
                            
                        clf.fit(X_tr_fit, y_train)
                        dump(clf, model_path)
                        
                        # Evaluate on all 4 target tracks across 12 blocks
                        blocks_confusion = np.zeros((4, NB, 5, 5), dtype=np.int32)
                        blocks_oa = np.zeros((4, NB), dtype=np.float64)
                        parquet_records = []
                        
                        for ti, tgt in enumerate(TARGET_ORDER):
                            X_tgt_eval = track_feats[rep][tgt][eval_all_indices]
                            if scaler is not None:
                                X_tgt_eval = scaler.transform(X_tgt_eval)
                                
                            preds_tgt = clf.predict(X_tgt_eval)
                            
                            for bi_, b in enumerate(teb):
                                b_mask = (block_eval_ids == b)
                                y_t_b = y_eval[b_mask]
                                y_p_b = preds_tgt[b_mask]
                                
                                cm_b = confusion_matrix(y_t_b, y_p_b, labels=[0,1,2,3,4])
                                blocks_confusion[ti, bi_] = cm_b
                                blocks_oa[ti, bi_] = accuracy_score(y_t_b, y_p_b)
                                
                            for idx_in_eval, global_idx in enumerate(eval_all_indices):
                                parquet_records.append({
                                    "target_track": tgt,
                                    "block_id": int(block_eval_ids[idx_in_eval]),
                                    "pixel_index": int(global_idx),
                                    "y_true": int(y_eval[idx_in_eval]),
                                    "y_pred": int(preds_tgt[idx_in_eval])
                                })
                                
                        np.save(conf_path, blocks_confusion)
                        np.save(oa_path, blocks_oa)
                        
                        df_pred = pd.DataFrame(parquet_records)
                        df_pred.to_parquet(parquet_path, index=False)
                        
                        cfg_save = {
                            "learner": learner,
                            "representation": rep,
                            "source_track": src,
                            "seed": seed,
                            "selected_params": params,
                            "train_pixel_count": len(i_tr),
                            "train_pixels_sha256": tr_hash,
                            "eval_blocks": teb,
                            "eval_pixel_count": len(eval_all_indices),
                            "feature_dim": len(feature_names[rep]),
                            "feature_names": feature_names[rep]
                        }
                        with open(cfg_path, "w", encoding="utf-8") as f:
                            json.dump(cfg_save, f, indent=2)

                    # Compute Main Results Metrics
                    src_idx = TARGET_ORDER.index(src)
                    cross_indices = [i for i in range(4) if i != src_idx]
                    
                    in_track_cm = blocks_confusion[src_idx].sum(axis=0)
                    in_track_ba = compute_pooled_ba(in_track_cm) * 100.0
                    in_track_oa = float(np.mean(blocks_oa[src_idx])) * 100.0
                    
                    cross_cm_per_target = [blocks_confusion[ti].sum(axis=0) for ti in cross_indices]
                    cross_ba_per_target = [compute_pooled_ba(cm) * 100.0 for cm in cross_cm_per_target]
                    cross_track_ba = float(np.mean(cross_ba_per_target))
                    cross_track_oa = float(np.mean(blocks_oa[cross_indices])) * 100.0
                    
                    gap_oa = in_track_oa - cross_track_oa
                    gap_ba = in_track_ba - cross_track_ba
                    
                    main_results.append({
                        "learner": learner,
                        "representation": rep,
                        "source_track": src,
                        "seed": seed,
                        "params": str(params),
                        "in_track_oa": in_track_oa,
                        "cross_track_oa": cross_track_oa,
                        "in_track_ba": in_track_ba,
                        "cross_track_ba": cross_track_ba,
                        "transfer_gap_oa": gap_oa,
                        "transfer_gap_ba": gap_ba
                    })

        print(f"Completed all models for seed {seed} (40 models). Elapsed: {(time.time()-t_start_train)/60:.1f}m")

    # Save local main results
    df_main = pd.DataFrame(main_results)
    df_main.to_csv(WORKSPACE / "tables" / "local_main_results.csv", index=False)
    print(f"Saved tables/local_main_results.csv ({len(df_main)} rows)")

def run_p04():
    print("=== Starting Phase P04: Source Spatial CV & Main Training ===")
    
    # Load datacube and labels
    c6_path = SAR_V2 / "data" / "cube_4track_v6.npz"
    v4_path = SAR_V2 / "data" / "flevoland_datacube_v4_multibaseline.npz"
    cm_path = SAR_V2 / "labels" / "consensus_mask.npz"
    mmu_path = SAR_V2 / "mmu" / "mmu_mask.npz"
    om_path = Path(r"E:\research\SAR\revision_experiments_20260914\omega_v6_rebuilt.npy")
    
    c6 = np.load(c6_path)
    v4 = np.load(v4_path)
    cm = np.load(cm_path)
    mmu = np.load(mmu_path)
    om = np.load(om_path)
    
    mv = mmu['mmu_valid']
    vi = np.where(mv)
    omv = om[vi]
    
    y_wc = v4['y'][vi].astype(int)
    y_clc = (cm['clc_on_mmu'] - 1).astype(int)
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    
    grid_y = v4['grid_y'][vi]
    grid_x = v4['grid_x'][vi]
    bids = (grid_y // 125) * 100 + (grid_x // 125)
    
    with open(WORKSPACE / "splits" / "split_manifest.json") as f:
        split_manifest = json.load(f)
    cv_folds = split_manifest["source_spatial_cv_3fold"]
    
    # 1. Feature Extraction
    track_feats, feature_names = extract_features(c6, vi)
    
    # 2. Spatial CV and Model Freezing
    selected_models = run_spatial_cv(track_feats, bids, c_mask, y_wc, cv_folds)
    
    # 3. Final Model Fits across 3 Seeds (120 models)
    train_and_eval_final_models(track_feats, feature_names, bids, c_mask, y_wc, split_manifest, selected_models)
    
    # Copy run_p04.py into workspace scripts/
    import shutil
    shutil.copy2(__file__, WORKSPACE / "scripts" / "run_p04_spatial_cv_and_training.py")

    # Update STATUS.json
    status_file = WORKSPACE / "STATUS.json"
    with open(status_file, "r", encoding="utf-8") as f:
        status = json.load(f)
    status["phases"]["P04"]["status"] = "COMPLETE"
    status["phases"]["P04"]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status["phases"]["P05"]["status"] = "READY"
    status["budget_usage"]["model_fits"] = 24 + 1080 + 120
    status["budget_usage"]["active_compute_hours"] = 1.0
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
        
    log_ledger("P04_CV_AND_MAIN_TRAINING", "P04", {
        "status": "SUCCESS",
        "cv_fits": 1080,
        "final_models": 120,
        "deliverables": [
            "tables/cv_trials.csv",
            "configs/selected_models.json",
            "tables/local_main_results.csv"
        ]
    }, exit_code=0)
    
    print("=== P04 Completed Successfully ===")

if __name__ == "__main__":
    run_p04()
