"""Deep spatial-context baseline for A0: small CNN on 9x9 patches, with and without coherence.

Question: can a model that learns spatial context from intensity patches replace the coherence
scalar? Arms (same training pixels, evaluation pixels, seeds and 400 m buffer as R1):
    CNN_I  : channels vv1, vh1, vv2, vh2 (dB, gamma0 v7 cube)
    CNN_IG : CNN_I + g12
Patch 9x9 at 40 m (half-width 160 m): train/eval windows cannot overlap under the 400 m buffer.
Training is fixed in advance (no test-based selection): 30 epochs, AdamW lr 1e-3, wd 1e-4,
batch 256, cross-entropy, no augmentation, last epoch used. Per-channel standardisation is fit on
the source-track training patches only. CPU (the GPU is used by other work).
Out: ../ws/runs_cnn/<arm>_<src>_s<seed>/blocks_confusion.npy (4 targets x 12 blocks x 5 x 5),
     blocks_confusion_full.npy (all C2 pixels of the eval blocks), config.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_run_R1_v7 as s4  # noqa: E402

r1 = s4.r1
WS = s4.WS
OUT = WS / "runs_cnn"
ARMS = {"CNN_I": ["d1_vv", "d1_vh", "d2_vv", "d2_vh"], "CNN_IG": ["d1_vv", "d1_vh", "d2_vv", "d2_vh", "g12"]}
HALF = 4
EPOCHS, LR, WD, BS = 30, 1e-3, 1e-4, 256
torch.set_num_threads(12)


class Net(nn.Module):
    def __init__(self, c):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(blk(c, 32), blk(32, 64), blk(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Linear(64, 5)

    def forward(self, x):
        return self.h(self.f(x))


def patches(stack, rows, cols):
    """stack: (C, H, W) with invalid = nan. Returns (N, C, 9, 9) float32 (nan kept)."""
    C, H, W = stack.shape
    pad = np.pad(stack, ((0, 0), (HALF, HALF), (HALF, HALF)), constant_values=np.nan)
    off = np.arange(-HALF, HALF + 1)
    rr = rows[:, None, None] + HALF + off[None, :, None]
    cc = cols[:, None, None] + HALF + off[None, None, :]
    return np.transpose(pad[:, rr, cc], (1, 0, 2, 3)).astype("float32")


def predict(model, X, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs])).argmax(1).numpy())
    return np.concatenate(out)


def block_cm(y, p, b, teb):
    cm = np.zeros((len(teb), 5, 5), dtype=np.int64)
    for bi, blk in enumerate(teb):
        m = b == blk
        np.add.at(cm[bi], (y[m], p[m]), 1)
    return cm


def main():
    sv = r1.SAR_V2
    c7 = np.load(s4.CUBE)
    v4 = np.load(sv / "data" / "flevoland_datacube_v4_multibaseline.npz")
    cm = np.load(sv / "labels" / "consensus_mask.npz")
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om = np.load(s4.SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy")
    vi = np.where(mmu["mmu_valid"])
    omv = om[vi]
    y_wc = v4["y"][vi].astype(int)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)
    gy, gx = v4["grid_y"][vi].astype(int), v4["grid_x"][vi].astype(int)
    bids = (gy // 125) * 100 + (gx // 125)
    coords = np.column_stack([gy, gx]).astype(float) * 40.0
    split = json.loads((WS / "splits" / "split_manifest.json").read_text())["historical_development_split"]
    trb, teb = split["train_blocks"], split["eval_blocks"]

    # evaluation pixels: identical to R1 (800 per block, RandomState(42 + b))
    ev = np.concatenate([np.random.RandomState(42 + b).choice(np.where((bids == b) & c_mask)[0],
                         min(800, int(((bids == b) & c_mask).sum())), replace=False) for b in teb])
    ev_full = np.where(np.isin(bids, teb) & c_mask)[0]
    tr_cand = np.where(np.isin(bids, trb) & c_mask)[0]
    eval_cand = np.where(np.isin(bids, teb) & c_mask)[0]
    d, _ = cKDTree(coords[eval_cand]).query(coords[tr_cand])
    tr_buf = tr_cand[d >= 400.0]

    valid = c7["four_track_valid"]
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, layers in ARMS.items():
        stacks = {t: np.stack([np.where(valid, c7[f"{t}_{k}"], np.nan) for k in layers]) for t in r1.TR}
        for seed in r1.SEEDS:
            i_tr = np.random.RandomState(seed).choice(tr_buf, min(20000, len(tr_buf)), replace=False)
            assert cKDTree(coords[i_tr]).query(coords[ev])[0].min() >= 400.0
            for src in r1.TR:
                name = f"{arm}_{src}_s{seed}"
                od = OUT / name
                if (od / "blocks_confusion_full.npy").exists():
                    continue
                t0 = time.time()
                Xtr = patches(stacks[src], gy[i_tr], gx[i_tr])
                mu = np.nanmean(Xtr, axis=(0, 2, 3), keepdims=True)
                sd = np.nanstd(Xtr, axis=(0, 2, 3), keepdims=True) + 1e-6

                def norm(X):
                    return np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")

                torch.manual_seed(seed)
                model = Net(len(layers))
                opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
                Xt, yt = torch.from_numpy(norm(Xtr)), torch.from_numpy(y_wc[i_tr].astype(np.int64))
                g = torch.Generator().manual_seed(seed)
                for ep in range(EPOCHS):
                    model.train()
                    perm = torch.randperm(len(Xt), generator=g)
                    for i in range(0, len(Xt), BS):
                        j = perm[i:i + BS]
                        opt.zero_grad()
                        nn.functional.cross_entropy(model(Xt[j]), yt[j]).backward()
                        opt.step()
                bc = np.zeros((4, len(teb), 5, 5), dtype=np.int64)
                bcf = np.zeros_like(bc)
                for ti, tgt in enumerate(r1.TARGET_ORDER):
                    p = predict(model, norm(patches(stacks[tgt], gy[ev], gx[ev])))
                    bc[ti] = block_cm(y_wc[ev], p, bids[ev], teb)
                    pf = predict(model, norm(patches(stacks[tgt], gy[ev_full], gx[ev_full])))
                    bcf[ti] = block_cm(y_wc[ev_full], pf, bids[ev_full], teb)
                od.mkdir(exist_ok=True)
                np.save(od / "blocks_confusion.npy", bc)
                np.save(od / "blocks_confusion_full.npy", bcf)
                torch.save(model.state_dict(), od / "model.pt")
                (od / "config.json").write_text(json.dumps({
                    "arm": arm, "layers": layers, "source_track": src, "seed": seed, "patch": 2 * HALF + 1,
                    "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "batch": BS,
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "train_pixels": int(len(i_tr)), "eval_pixels": int(len(ev)), "eval_pixels_full": int(len(ev_full)),
                    "seconds": round(time.time() - t0, 1)}, indent=2))
                print(f"{name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
