"""Site 2, Study 1: A0 R1 protocol (5 representations x RF/SVC x 4 source tracks x 3 seeds, 400 m buffer,
3-fold spatial CV for hyperparameters, 800 evaluation pixels per block) on the site-2 cube.

Reuses res1_clean/src/run_R1_buffered_cv_retrain.py functions unchanged; only the inputs differ
(site-2 cube, WorldCover/CORINE consensus, MMU, Omega, split manifest). The number of evaluation
blocks is read from the manifest (17 at site 2; the R1 code handles any number).
Out: ../runs/STUDY1/ws/{configs/R1_selected_models.json, runs_R1/*, provenance.json}
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "runs" / "STUDY1"
WS = S1 / "ws"
sys.path.insert(0, str(ROOT.parent / "res1_clean" / "src"))
import run_R1_buffered_cv_retrain as r1  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def inputs():
    """Arrays in the exact form R1 expects (flattened over mmu_valid, as in A0)."""
    c = dict(np.load(S1 / "cube_spring.npz"))
    L = np.load(S1 / "labels.npz")
    vi = np.where(L["mmu_valid"])
    omv = L["omega"][vi]
    y_wc = L["y"][vi].astype(int)
    y_clc = L["clc"][vi].astype(int) - 1
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    gy, gx = L["grid_y"][vi], L["grid_x"][vi]
    bids = (gy // 125) * 100 + (gx // 125)
    coords = np.column_stack([gy, gx]).astype(float) * 40.0
    return c, vi, y_wc, c_mask, bids, coords


def checkpointed_cv(feats, bids, c_mask, y_wc, folds, coords):
    """r1.run_spatial_cv, called once per (representation, source track) so that each finished cell is saved
    to configs/R1_cv_checkpoint.json; an interrupted run resumes after the last saved cell. The selection
    inside each cell is r1's code unchanged (it iterates over r1.REPS x r1.TR, restricted here to one cell)."""
    ck = WS / "configs" / "R1_cv_checkpoint.json"
    done = json.loads(ck.read_text()) if ck.exists() else {}
    reps, trs = list(r1.REPS), list(r1.TR)
    try:
        for rep in reps:
            for src in trs:
                if f"RF_{rep}_{src}" in done and f"SVC_{rep}_{src}" in done:
                    continue
                r1.REPS, r1.TR = [rep], [src]
                done.update(r1.run_spatial_cv(feats, bids, c_mask, y_wc, folds, coords))
                ck.write_text(json.dumps(done, indent=2))
                print(f"checkpoint {len(done)}/40", flush=True)
    finally:
        r1.REPS, r1.TR = reps, trs
    return done


def main():
    (WS / "splits").mkdir(parents=True, exist_ok=True)
    (WS / "configs").mkdir(exist_ok=True)
    shutil.copy2(S1 / "split_manifest.json", WS / "splits" / "split_manifest.json")
    r1.WORKSPACE = WS
    c, vi, y_wc, c_mask, bids, coords = inputs()
    sm = json.loads((WS / "splits" / "split_manifest.json").read_text())
    prov = {"cube_sha256": sha256(S1 / "cube_spring.npz"), "labels_sha256": sha256(S1 / "labels.npz"),
            "c2_pixels": int(c_mask.sum()), "eval_blocks": sm["historical_development_split"]["eval_blocks"],
            "r1_source_sha256": sha256(ROOT.parent / "res1_clean" / "src" / "run_R1_buffered_cv_retrain.py")}
    (WS / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(prov, flush=True)
    feats, names = r1.extract_features(c, vi)
    sel_path = WS / "configs" / "R1_selected_models.json"
    # r1.run_spatial_cv rewrites this file after every cell, so only a 40-entry file counts as complete
    prev = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if len(prev) == 40:
        selected = prev
    else:
        selected = checkpointed_cv(feats, bids, c_mask, y_wc, sm["source_spatial_cv_3fold"], coords)
        sel_path.write_text(json.dumps(selected, indent=2))
    try:
        r1.train_and_eval_final_models(feats, names, bids, c_mask, y_wc, sm, selected, coords)
    except KeyError as exc:
        print(f"final models done; skipped legacy old-vs-new step ({exc!r})")


if __name__ == "__main__":
    main()
