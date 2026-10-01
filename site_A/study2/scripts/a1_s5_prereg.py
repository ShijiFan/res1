"""Step 5: freeze the pre-registered analysis plan BEFORE any classifier is trained on the main data.

Writes RUN/reports/A1_PREREG.md and RUN/reports/A1_PREREG.sha256. a1_s6_experiments.py refuses to run
without them, and records the hash in every output. Re-running this script after experiments have
started is refused (the plan must not change after seeing results).
"""
from __future__ import annotations

import hashlib

from a1_common import BUDGETS, D, SEED, TARGET_CLASSES, ensure_dirs, now, update_status

PLAN = f"""# A1 pre-registered analysis plan (frozen {now()})

## Data and design
- Site: Twente (NL), common 40 m grid EPSG:32632 (runs/20260923_G0L_COMMON_GRID/grid_spec.json).
- Years: 2025 (primary), 2024 (replication). Tracks: T37 (descending), T88 (ascending).
- Budget M = number of disjoint 12-day S1A/S1A interferometric pairs; scenes N = 2M.
  M in {list(BUDGETS)}. Uniform design = budget_design.csv (branch MAIN_12D_AA): pairs spread over
  March-October; nested for M >= 2; M = 1 is the mid-season (June) pair. A design pair lost to a failed
  HyP3 job is replaced by the temporally closest available pair (logged in experiments/REGISTRY.csv).
  Random designs (E3): M pairs drawn from the 9 available pairs (20 draws in 2025, 10 in 2024).
- Labels: BRP (Definitief) of the same year; classes {TARGET_CLASSES}; strict season mapping
  (G0L_SEMANTIC_DECISION.md); class "Other" excluded from training and evaluation in the primary analysis.
- Unit: parcel (mean of pure 40 m pixels, >= 2 pure pixels). Intensity averaged in linear power, then dB.
- Splits: spatial_splits_{{year}}.csv, 6 folds of 5 km blocks, 400 m exclusion buffer, seed {SEED}.

## Observables
- O1: VV dB at the 2M scenes. O2: VV, VH, VH-VV dB. O3: O2 + 12-day VV coherence of the M pairs. O5: coherence only.

## Learners
- Primary: HistGradientBoosting (fixed hyper-parameters, no tuning, class_weight=balanced).
- Robustness: RandomForest, multinomial logistic regression (M in 1, 2, 4, 9).

## Metrics and statistics
- Primary metric: balanced accuracy (BA) over the 7 target classes, pooled out-of-fold predictions.
- Secondary: macro-F1, per-class recall and F1, overall accuracy.
- Uncertainty: bootstrap over 5 km test blocks (B = 10000 for headline contrasts, 2000 otherwise),
  identical resampling weights for all configurations of the same test year (paired).
- Coherence gain: dC(M) = BA(O3, M) - BA(O2, M). Two-sided bootstrap p-values, Holm across the 7 budgets.
- Absorption point N*: smallest M from which the upper 95% bound of dC stays < 1 pp for every larger budget
  (reported as M and N = 2M; the first M below 1 pp is reported alongside).
- Equivalent-scene multiplier: m(M) = M'/M where BA(O2, M') = BA(O3, M), linear interpolation on the O2 curve.
- Equal-cost contrast: credits = 15 per RTC scene and 15 per pair, so cost(O2, M) = 30M and cost(O3, M) = 45M;
  compare O3 at M with O2 at 1.5M for M = 2, 4, 6.

## Hypotheses (decided before seeing results)
- H1: dC(M) decreases with M; dC(1) >= 3 pp in 2025 for at least one track. Falsified if the 95% upper bound of dC(1) < 3 pp for both tracks.
- H2: N* <= 8 pairs (16 scenes) in 2025 for both tracks. Falsified if dC(9) has lower 95% bound > 1 pp.
- H3: at M >= N*, TempGrass and Maize keep a positive F1 gain (lower 95% bound > 0). Falsified otherwise.
- H4: in the 2025 T88 windows, adding 6-day coherence to 12-day coherence raises BA (lower bound > 0). Falsified otherwise.
- H5: |N*(2025) - N*(2024)| <= 2 pairs for each track. Falsified otherwise.
- Q6 (open): at equal total pairs, two tracks (M/2 each) vs one track (M). No direction predicted.
Every outcome, including falsification, is reported.

## Controls
- Label permutation (20 repetitions), test-time date-order permutation, track-identity probe.
"""


def main() -> None:
    ensure_dirs()
    md = D["reports"] / "A1_PREREG.md"
    if md.exists() and any(D["exp"].glob("*/*.npz")):
        raise SystemExit("STOP: experiments already started; the pre-registration must not be changed")
    md.write_text(PLAN, encoding="utf-8")
    h = hashlib.sha256(PLAN.encode("utf-8")).hexdigest()
    (D["reports"] / "A1_PREREG.sha256").write_text(h + "\n", encoding="utf-8")
    update_status("s5_prereg", "DONE", sha256=h)
    print(h)


if __name__ == "__main__":
    main()
