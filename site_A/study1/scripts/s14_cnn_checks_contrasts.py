"""Block-bootstrap contrasts for the CNN supplementary checks (PROTOCOL.md 5.3).

Primary: tuned_p9 (C1 hyper-parameters, 9x9, 5 seeds) CNN_IG - CNN_I. Secondary: fixed_p{5,9,11,15}.
Also tuned CNN_I vs RF_I28 (RF uses its 3 seeds). Same bootstrap as s11 / T1.
Out: ../ws/tables_cnn/contrasts_checks.csv
"""
import numpy as np
import pandas as pd

from s11_contrasts import WS, N_BOOT, load_family, summarise

SEEDS5 = [17, 29, 43, 53, 71]


def main():
    idx = np.random.RandomState(42).randint(0, 12, size=(N_BOOT, 12))
    counts = np.eye(12)[idx].sum(axis=1)
    point = np.ones((1, 12))
    rows = []
    fams = [("tuned_p9", "primary"), ("fixed_p5", "C3"), ("fixed_p9", "C3"), ("fixed_p11", "C3"), ("fixed_p15", "C3 (600 m buffer)")]
    rf = load_family("runs_R1", "RF_I28", "blocks_confusion.npy")
    for tag, role in fams:
        for fname, ev in (("blocks_confusion.npy", "sample"), ("blocks_confusion_full.npy", "full")):
            a = load_family(f"runs_cnn2/{tag}", "CNN_IG", fname, SEEDS5)
            b = load_family(f"runs_cnn2/{tag}", "CNN_I", fname, SEEDS5)
            pairs = [("CNN_IG - CNN_I", a, b)]
            if ev == "sample":
                pairs.append(("CNN_I - RF_I28", b, rf))
            for name, A, B in pairs:
                pa, pia, _ = summarise(A, point)
                pb, pib, _ = summarise(B, point)
                ba_a, _, oka = summarise(A, counts)
                ba_b, _, okb = summarise(B, counts)
                ok = oka & okb
                d = (ba_a - ba_b)[ok]
                rows.append({"family": tag, "role": role, "eval": ev, "contrast": name,
                             "A_cross_ba": round(float(pa[0]), 3), "B_cross_ba": round(float(pb[0]), 3),
                             "delta_cross_ba": round(float(pa[0] - pb[0]), 3),
                             "ci_lo": round(float(np.percentile(d, 2.5)), 3), "ci_hi": round(float(np.percentile(d, 97.5)), 3),
                             "A_in_ba": round(float(pia[0]), 3), "B_in_ba": round(float(pib[0]), 3), "n_valid_boot": int(ok.sum())})
    df = pd.DataFrame(rows)
    (WS / "tables_cnn").mkdir(parents=True, exist_ok=True)
    df.to_csv(WS / "tables_cnn" / "contrasts_checks.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
