"""
Bootstrap Sensitivity Audit (Phase 3B):
Compares standard paired block bootstrap (20,000 replicates) against
class-support-preserving block bootstrap (resampling until all 5 classes are present).
Covers RF and SVC, cross-track Balanced Accuracy and Overall Accuracy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

RUNS_R1 = Path(r"E:\research\SAR\tgrs_final_campaign_20260915\runs_R1")
OUT_DIR = Path(r"E:\research\SAR\paper\tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 20000
SEEDS = [17, 29, 43]
TR = ["t15", "t37", "t88", "t139"]
CLASSES = ["Forest", "Grassland", "Cropland", "Urban", "Water"]

def compute_ba(cm):
    rec = [cm[k, k] / cm[k].sum() for k in range(5) if cm[k].sum() > 0]
    return float(np.mean(rec)) * 100.0 if rec else 0.0

def compute_oa(cm):
    return float(np.trace(cm) / cm.sum()) * 100.0 if cm.sum() > 0 else 0.0

def main():
    print("=== Running Bootstrap Sensitivity Audit (Phase 3B) ===")
    
    # Load all block confusion matrices from runs_R1
    # runs_R1/{learner}_{rep}_{src}_s{seed}/blocks_confusion.npy
    # shape: (4 target tracks, 12 blocks, 5, 5)
    
    cms_by_model = {}
    for learner in ["RF", "SVC"]:
        for rep in ["I8", "I28", "I8_G9", "F12", "I28_G29"]:
            block_cms = {b: np.zeros((5, 5), dtype=int) for b in range(12)}
            for src in TR:
                for seed in SEEDS:
                    p = RUNS_R1 / f"{learner}_{rep}_{src}_s{seed}" / "blocks_confusion.npy"
                    if not p.exists():
                        continue
                    arr = np.load(p) # (4, 12, 5, 5)
                    # Sum over cross-track target tracks
                    src_idx = TR.index(src)
                    for tgt_idx in range(4):
                        if tgt_idx == src_idx:
                            continue # cross-track only
                        for b in range(12):
                            block_cms[b] += arr[tgt_idx, b]
            cms_by_model[(learner, rep)] = block_cms
            
    print("Pre-aggregated block confusion matrices.")
    np.random.seed(42)
    sample_idx = np.random.randint(0, 12, size=(N_BOOT, 12))
    
    results = []
    
    for learner in ["RF", "SVC"]:
        base_cm = cms_by_model[(learner, "I28")]
        aug_cm = cms_by_model[(learner, "I28_G29")]
        
        # 1. Standard Paired Block Bootstrap
        std_delta_ba = []
        std_delta_oa = []
        n_missing_classes = 0
        
        # 2. Class-Support Preserving Bootstrap
        csp_delta_ba = []
        csp_delta_oa = []
        
        for i in range(N_BOOT):
            sel = sample_idx[i]
            cb = sum(base_cm[b] for b in sel)
            cg = sum(aug_cm[b] for b in sel)
            
            # Check if any class has 0 true support in this replicate
            has_missing = np.any(cb.sum(axis=1) == 0) or np.any(cg.sum(axis=1) == 0)
            if has_missing:
                n_missing_classes += 1
                
            ba_b = compute_ba(cb)
            ba_g = compute_ba(cg)
            oa_b = compute_oa(cb)
            oa_g = compute_oa(cg)
            
            std_delta_ba.append(ba_g - ba_b)
            std_delta_oa.append(oa_g - oa_b)
            
            if not has_missing:
                csp_delta_ba.append(ba_g - ba_b)
                csp_delta_oa.append(oa_g - oa_b)
                
        # Compute summary stats
        std_ci_ba = np.percentile(std_delta_ba, [2.5, 97.5])
        std_ci_oa = np.percentile(std_delta_oa, [2.5, 97.5])
        
        csp_ci_ba = np.percentile(csp_delta_ba, [2.5, 97.5])
        csp_ci_oa = np.percentile(csp_delta_oa, [2.5, 97.5])
        
        results.append({
            "learner": learner,
            "metric": "Balanced Accuracy (BA)",
            "contrast": "I28_G29 - I28",
            "bootstrap_method": "Standard Paired Block Bootstrap",
            "mean_gain": round(float(np.mean(std_delta_ba)), 2),
            "ci_95_lo": round(float(std_ci_ba[0]), 2),
            "ci_95_hi": round(float(std_ci_ba[1]), 2),
            "ci_width": round(float(std_ci_ba[1] - std_ci_ba[0]), 2),
            "n_replicates": N_BOOT,
            "replicates_missing_class": n_missing_classes,
            "missing_rate_pct": round(n_missing_classes / N_BOOT * 100.0, 2)
        })
        results.append({
            "learner": learner,
            "metric": "Balanced Accuracy (BA)",
            "contrast": "I28_G29 - I28",
            "bootstrap_method": "Class-Support Preserving Bootstrap",
            "mean_gain": round(float(np.mean(csp_delta_ba)), 2),
            "ci_95_lo": round(float(csp_ci_ba[0]), 2),
            "ci_95_hi": round(float(csp_ci_ba[1]), 2),
            "ci_width": round(float(csp_ci_ba[1] - csp_ci_ba[0]), 2),
            "n_replicates": len(csp_delta_ba),
            "replicates_missing_class": 0,
            "missing_rate_pct": 0.0
        })
        results.append({
            "learner": learner,
            "metric": "Overall Accuracy (OA)",
            "contrast": "I28_G29 - I28",
            "bootstrap_method": "Standard Paired Block Bootstrap",
            "mean_gain": round(float(np.mean(std_delta_oa)), 2),
            "ci_95_lo": round(float(std_ci_oa[0]), 2),
            "ci_95_hi": round(float(std_ci_oa[1]), 2),
            "ci_width": round(float(std_ci_oa[1] - std_ci_oa[0]), 2),
            "n_replicates": N_BOOT,
            "replicates_missing_class": n_missing_classes,
            "missing_rate_pct": round(n_missing_classes / N_BOOT * 100.0, 2)
        })
        results.append({
            "learner": learner,
            "metric": "Overall Accuracy (OA)",
            "contrast": "I28_G29 - I28",
            "bootstrap_method": "Class-Support Preserving Bootstrap",
            "mean_gain": round(float(np.mean(csp_delta_oa)), 2),
            "ci_95_lo": round(float(csp_ci_oa[0]), 2),
            "ci_95_hi": round(float(csp_ci_oa[1]), 2),
            "ci_width": round(float(csp_ci_oa[1] - csp_ci_oa[0]), 2),
            "n_replicates": len(csp_delta_oa),
            "replicates_missing_class": 0,
            "missing_rate_pct": 0.0
        })

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "bootstrap_sensitivity.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    print(df[["learner", "metric", "bootstrap_method", "mean_gain", "ci_95_lo", "ci_95_hi", "missing_rate_pct"]].to_string())

if __name__ == "__main__":
    main()
