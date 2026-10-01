"""A1 G0: HyP3 credit balance and real per-job costs.

Credentials are read from environment variables (never hard-code them):
    set EARTHDATA_USERNAME=...      (Windows cmd)   /  $env:EARTHDATA_USERNAME="..." (PowerShell)
    set EARTHDATA_PASSWORD=...
or leave them unset to fall back to ~/.netrc (machine urs.earthdata.nasa.gov).

    pip install hyp3_sdk
    python g0_hyp3_check.py --n-rtc 80 --n-insar 166
"""
import argparse, json, os

import hyp3_sdk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rtc", type=int, default=80)
    ap.add_argument("--n-insar", type=int, default=166)
    a = ap.parse_args()

    user, pw = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    hyp3 = hyp3_sdk.HyP3(username=user, password=pw) if user and pw else hyp3_sdk.HyP3()

    credits = hyp3.check_credits()
    print("remaining credits:", credits)

    try:
        costs = hyp3.costs()  # available in recent hyp3_sdk versions
        print(json.dumps(costs, indent=1)[:3000])
    except Exception as e:  # older SDK: read the cost table on the HyP3 docs page instead
        costs = None
        print("hyp3.costs() not available:", e)

    print("\nFill in the unit costs you see above, then:")
    for c_rtc in (1, 5, 15, 60):
        for c_ins in (10, 15):
            tot = a.n_rtc * c_rtc + a.n_insar * c_ins
            print(f"  c_RTC={c_rtc:>2}  c_InSAR={c_ins:>2}  -> total {tot:>6}  "
                  f"({'OK' if credits and tot * 1.3 <= credits else 'check'})")


if __name__ == "__main__":
    main()
