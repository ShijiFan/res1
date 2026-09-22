"""
Calculate Spring 12-block paired sign test for runs_R1 (Headline Protocol).
Compares I28_G29 vs I28 across 12 evaluation blocks under cross-track evaluation.
Saves results to tables/spring_block_sign_test.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

WORKSPACE = Path(r"E:\research\SAR\tgrs_final_campaign_20260915")
RUNS_NEW = WORKSPACE / "runs_R1"
TABLES_DIR = WORKSPACE / "tables"
TABLES_DIR.mkdir(exist_ok=True)

TR = ["t15", "t37", "t88", "t139"]
SEEDS = [17, 29, 43]
CLASSES = ["Forest", "Grassland", "Cropland", "Urban", "Water"]

conf = {l: {r: {s: {} for s in TR} for r in ["I28", "I28_G29"]} for l in ["RF", "SVC"]}

for l in ["RF", "SVC"]:
    for r in ["I28", "I28_G29"]:
        for s in TR:
            for seed in SEEDS:
                p = RUNS_NEW / f"{l}_{r}_{s}_s{seed}" / "blocks_confusion.npy"
                conf[l][r][s][seed] = np.load(p)

records = []
summary_records = []

for l in ["RF", "SVC"]:
    cm_base = np.zeros((12, 5, 5))
    cm_aug  = np.zeros((12, 5, 5))
    
    for b in range(12):
        for seed in SEEDS:
            for si, src in enumerate(TR):
                for ti, tgt in enumerate(TR):
                    if si == ti:
                        continue # cross-track only
                    cm_base[b] += conf[l]["I28"][src][seed][ti, b]
                    cm_aug[b]  += conf[l]["I28_G29"][src][seed][ti, b]

    supp_base = cm_base.sum(axis=2) # (12, 5)
    
    diffs = []
    for b in range(12):
        s = supp_base[b]
        present = s > 0
        rec_b = np.diag(cm_base[b])[present] / s[present] * 100.0
        rec_a = np.diag(cm_aug[b])[present] / s[present] * 100.0
        ba_b = float(rec_b.mean())
        ba_a = float(rec_a.mean())
        dba = ba_a - ba_b
        diffs.append(dba)
        
        records.append({
            "season": "spring",
            "learner": l,
            "block_id": b,
            "ba_I28": round(ba_b, 4),
            "ba_I28_g12": round(ba_a, 4),
            "delta_ba": round(dba, 4),
            "is_positive": dba > 0
        })
        
    diffs = np.array(diffs)
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    zero = int(np.sum(diffs == 0))
    bt = binomtest(pos, 12, 0.5, alternative="two-sided")
    
    summary_records.append({
        "learner": l,
        "n_blocks": 12,
        "n_positive": pos,
        "n_negative": neg,
        "n_zero": zero,
        "min_delta_ba": round(float(diffs.min()), 2),
        "max_delta_ba": round(float(diffs.max()), 2),
        "mean_delta_ba": round(float(diffs.mean()), 2),
        "median_delta_ba": round(float(np.median(diffs)), 2),
        "p_value": round(float(bt.pvalue), 6)
    })

df_blocks = pd.DataFrame(records)
df_summary = pd.DataFrame(summary_records)

out_blocks = TABLES_DIR / "spring_block_sign_test.csv"
out_summary = TABLES_DIR / "spring_block_sign_test_summary.csv"

df_blocks.to_csv(out_blocks, index=False)
df_summary.to_csv(out_summary, index=False)

print(f"Saved block test details to {out_blocks}")
print(f"Saved block test summary to {out_summary}")
print("\nSummary:")
print(df_summary.to_string(index=False))
