"""Site 2, Study 1: CNN with the site-1 pre-registered protocol (PREREG_SITE2.md section 2).

Reuses A0_gamma0_rerun_20260929/scripts/s13_cnn_checks.py unchanged (network, patches, training, CV grid,
selection rule, 5 seeds, 9x9 patches, 400 m buffer) and replaces only its data loader and output folders
with site-2 inputs. Hyper-parameters are selected by 3-fold spatial CV inside the SITE-2 training blocks.
Usage: python s2_16_study1_cnn.py --stage cv | final
Out:   ../runs/STUDY1/ws/cnn_checks/cv_selected.json, ../runs/STUDY1/ws/runs_cnn2/tuned_p9/*
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "runs" / "STUDY1"
WS = S1 / "ws"
sys.path.insert(0, str(ROOT.parent / "A0_gamma0_rerun_20260929" / "scripts"))
import s13_cnn_checks as s13  # noqa: E402
from s10_cnn_baseline import ARMS  # noqa: E402


def load_site2():
    c7 = np.load(S1 / "cube_spring.npz")
    L = np.load(S1 / "labels.npz")
    vi = np.where(L["mmu_valid"])
    omv = L["omega"][vi]
    y = L["y"][vi].astype(np.int64)
    y_clc = L["clc"][vi].astype(int) - 1
    c_mask = omv & (y >= 0) & (y < 5) & (y_clc >= 0) & (y_clc < 5) & (y == y_clc)
    gy, gx = L["grid_y"][vi].astype(int), L["grid_x"][vi].astype(int)
    bids = (gy // 125) * 100 + (gx // 125)
    coords = np.column_stack([gy, gx]).astype(float) * 40.0
    sm = json.loads((S1 / "split_manifest.json").read_text())
    valid = c7["four_track_valid"]
    stacks = {arm: {t: np.stack([np.where(valid, c7[f"{t}_{k}"], np.nan) for k in layers]) for t in s13.r1.TR}
              for arm, layers in ARMS.items()}
    return dict(y=y, c_mask=c_mask, gy=gy, gx=gx, bids=bids, coords=coords, sm=sm, stacks=stacks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "final"], required=True)
    a = ap.parse_args()
    s13.OUT = WS / "cnn_checks"
    s13.RUNS = WS / "runs_cnn2"
    D = load_site2()
    if a.stage == "cv":
        s13.stage_cv(D)
    else:
        sel = json.loads((s13.OUT / "cv_selected.json").read_text())
        s13.run_family(D, "tuned_p9", 4, lambda arm, src: {k: sel[f"{arm}_{src}"][k] for k in ("lr", "epochs", "wd")}, 400.0)


if __name__ == "__main__":
    main()
