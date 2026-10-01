"""Site 2, Study 1: 5 km block split in the A0 split_manifest.json format (PREREG_SITE2.md section 1, D1).

  * blocks: bids = (grid_y // 125) * 100 + grid_x // 125 (as A0);
  * eligible blocks: Omega / MMU-valid cells > 0.90 (A0 rule); at site 2 Omega == MMU by construction,
    so every block with MMU cells is eligible (reported);
  * evaluation blocks: 30% drawn with seed 20260930; every class must occur in >= 3 evaluation blocks
    (counted on the consensus set C2); otherwise seed + 1, + 2, ... (logged);
  * 3-fold CV inside the training blocks: contiguous block rows split into north / central / south thirds
    (A0 practice).
Out: ../runs/STUDY1/split_manifest.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "runs" / "STUDY1"


def main():
    L = np.load(S1 / "labels.npz")
    mv, om, y, clc = L["mmu_valid"], L["omega"], L["y"].astype(int), L["clc"].astype(int)
    gy, gx = L["grid_y"], L["grid_x"]
    bids = (gy // 125) * 100 + gx // 125
    c2 = om & (y >= 0) & (clc > 0) & (clc - 1 == y)
    blocks = sorted(int(b) for b in np.unique(bids[mv]))
    elig = [b for b in blocks if om[bids == b].sum() / max(1, mv[bids == b].sum()) > 0.90]
    presence = {b: set(np.unique(y[c2 & (bids == b)]).tolist()) for b in elig}
    n_eval = int(round(0.3 * len(elig)))
    seed, tries = 20260930, []
    while True:
        perm = np.random.RandomState(seed).permutation(elig)
        ev = sorted(int(b) for b in perm[:n_eval])
        cnt = {c: sum(c in presence[b] for b in ev) for c in range(5)}
        tries.append({"seed": seed, "class_block_counts": cnt})
        if all(v >= 3 for v in cnt.values()):
            break
        seed += 1
    tr = sorted(b for b in elig if b not in ev)
    rows = sorted({b // 100 for b in tr})
    thirds = np.array_split(rows, 3)
    folds = {f"fold_{i + 1}_blocks": [b for b in tr if b // 100 in set(t.tolist())] for i, t in enumerate(thirds)}
    man = {"grid_specification": {"block_px": 125, "block_m": 5000, "bids_rule": "(grid_y//125)*100 + grid_x//125"},
           "historical_development_split": {"train_blocks": tr, "eval_blocks": ev, "seed_used": seed,
                                            "seed_attempts": tries, "eligible_blocks": elig},
           "source_spatial_cv_3fold": {**folds, "fold_rows": [t.tolist() for t in thirds],
                                       "rule": "contiguous block rows, north/central/south thirds"}}
    (S1 / "split_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"eligible {len(elig)} / {len(blocks)}; train {len(tr)}, eval {len(ev)} (seed {seed}); folds",
          {k: len(v) for k, v in folds.items()}, "; class presence in eval blocks", tries[-1]["class_block_counts"])


if __name__ == "__main__":
    main()
