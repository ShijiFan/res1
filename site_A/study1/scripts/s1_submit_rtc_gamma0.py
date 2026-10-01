"""Re-order spring-2024 RTC for the four A0 tracks as gamma0 power (HyP3 'gpuned').

Reason: A0 used sigma0-dB (sdunem, DEM matching) for T15/T37 and gamma0-power
(gpuned) for T88/T139. The autumn and A1 products are all gpuned. Re-ordering all
eight spring scenes with one parameter set removes the mixed radiometry.

Usage:  python s1_submit_rtc_gamma0.py --dry-run   (price + balance only)
        python s1_submit_rtc_gamma0.py --submit
Credentials: ~/.netrc (urs.earthdata.nasa.gov). Nothing is hard-coded.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import hyp3_sdk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "jobs"

# Source granules, read from the original product READMEs in sar_v2/data/products.
GRANULES = {
    "T139_t1": "S1A_IW_SLC__1SDV_20240507T054228_20240507T054255_053761_06884C_E633",
    "T139_t2": "S1A_IW_SLC__1SDV_20240519T054228_20240519T054255_053936_068E5E_1619",
    "T15_t1": "S1A_IW_SLC__1SDV_20240510T171724_20240510T171751_053812_068A1A_9006",
    "T15_t2": "S1A_IW_SLC__1SDV_20240522T171723_20240522T171750_053987_06901E_285A",
    "T37_t1": "S1A_IW_SLC__1SDV_20240512T055038_20240512T055105_053834_068ADC_3D7B",
    "T37_t2": "S1A_IW_SLC__1SDV_20240524T055038_20240524T055105_054009_0690DF_FB9B",
    "T88_t1": "S1A_IW_SLC__1SDV_20240515T172529_20240515T172556_053885_068C95_34DB",
    "T88_t2": "S1A_IW_SLC__1SDV_20240527T172529_20240527T172557_054060_0692AB_3D13",
}
# Same radiometry as autumn (RTC10 gpuned) and A1 (RTC20 gpuned); 20 m matches A1 (user choice 2026-09-29).
# All other options left at HyP3 defaults.
RTC_PARAMS = dict(radiometry="gamma0", scale="power", resolution=20)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    hyp3 = hyp3_sdk.HyP3()
    payloads = [hyp3_sdk.HyP3.prepare_rtc_job(gr, name=f"A0g0_{k}", **RTC_PARAMS)
                for k, gr in GRANULES.items()]
    balance = hyp3.check_credits()
    try:
        cost = sum(hyp3.costs()["RTC_GAMMA"]["cost_table"][str(RTC_PARAMS["resolution"])] for _ in payloads)
    except Exception as exc:  # cost table layout differs across API versions
        cost = f"unknown ({exc!r})"
    print(f"hyp3_sdk {hyp3_sdk.__version__}; balance={balance}; estimated cost={cost}")
    for p in payloads:
        print(json.dumps(p))
    if a.dry_run:
        return

    batch = hyp3.submit_prepared_jobs(payloads)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    rec = {"submitted_at": stamp, "hyp3_sdk": hyp3_sdk.__version__, "params": RTC_PARAMS,
           "balance_before": balance, "jobs": [j.to_dict() for j in batch]}
    (OUT / f"submitted_{stamp}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"submitted {len(batch)} jobs -> {OUT / f'submitted_{stamp}.json'}")


if __name__ == "__main__":
    main()
