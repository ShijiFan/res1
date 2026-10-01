"""Re-run the A0 R1 protocol (400 m buffered spatial CV + 120 final models) on cube_4track_v7.

Reuses res1_clean/src/run_R1_buffered_cv_retrain.py unchanged (same features, grids, folds,
seeds, buffer, evaluation sample). Only the cube path and the output workspace differ.
Hyperparameters are RE-SELECTED by the same 3-fold spatial CV, because the intensity
inputs changed; the published selection stays in tgrs_final_campaign_20260915/configs/.

Out: ../ws/configs/R1_selected_models.json, ../ws/runs_R1/<learner>_<rep>_<src>_s<seed>/
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAR = ROOT.parent
WS = ROOT / "ws"
import os
CUBE = Path(os.environ.get("A0_CUBE", ROOT / "data" / "cube_4track_v7.npz"))  # override only for smoke tests
SRC = SAR / "res1_clean" / "src"
CAMPAIGN = SAR / "tgrs_final_campaign_20260915"

sys.path.insert(0, str(SRC))
import run_R1_buffered_cv_retrain as r1  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    (WS / "splits").mkdir(parents=True, exist_ok=True)
    (WS / "configs").mkdir(exist_ok=True)
    shutil.copy2(CAMPAIGN / "splits" / "split_manifest.json", WS / "splits" / "split_manifest.json")
    r1.WORKSPACE = WS

    sv = r1.SAR_V2
    c7 = np.load(CUBE)
    v4 = np.load(sv / "data" / "flevoland_datacube_v4_multibaseline.npz")
    cm = np.load(sv / "labels" / "consensus_mask.npz")
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om_path = SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy"
    om = np.load(om_path)

    mv = mmu["mmu_valid"]
    vi = np.where(mv)
    omv = om[vi]
    # Omega stays the published support; count pixels where v7 intensity is missing on it.
    lost = int((omv & ~c7["four_track_valid"][vi]).sum())
    y_wc = v4["y"][vi].astype(int)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    grid_y, grid_x = v4["grid_y"][vi], v4["grid_x"][vi]
    bids = (grid_y // 125) * 100 + (grid_x // 125)
    coords = np.column_stack([grid_y, grid_x]).astype(float) * 40.0
    split_manifest = json.loads((WS / "splits" / "split_manifest.json").read_text())

    prov = {"cube": str(CUBE), "cube_sha256": sha256(CUBE), "omega": str(om_path),
            "omega_sha256": sha256(om_path), "omega_pixels_without_v7_intensity": lost,
            "c2_pixels": int(c_mask.sum()), "r1_source": str(SRC / "run_R1_buffered_cv_retrain.py"),
            "r1_source_sha256": sha256(SRC / "run_R1_buffered_cv_retrain.py")}
    (WS / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(prov)
    if lost > 0.001 * int(omv.sum()):
        raise SystemExit(f"STOP: {lost} Omega pixels lack v7 intensity (>0.1%); reconcile support first.")

    feats, names = r1.extract_features(c7, vi)
    sel_path = WS / "configs" / "R1_selected_models.json"
    if sel_path.exists():  # CV already completed on this cube (resume after a later failure)
        selected = json.loads(sel_path.read_text())
        print(f"reusing CV selection {sel_path} sha256={sha256(sel_path)}")
    else:
        selected = r1.run_spatial_cv(feats, bids, c_mask, y_wc, split_manifest["source_spatial_cv_3fold"], coords)
    try:
        r1.train_and_eval_final_models(feats, names, bids, c_mask, y_wc, split_manifest, selected, coords)
    except KeyError as exc:
        # Step 3 of the original compares against WORKSPACE/runs (pre-buffer runs), absent here.
        print(f"final models done; skipped legacy old-vs-new step ({exc!r})")


if __name__ == "__main__":
    main()
