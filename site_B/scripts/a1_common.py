"""Shared paths, constants and helpers for the A1 main pipeline (steps a1_s1 ... a1_s9).

All scripts resolve ROOT from their own location, so the folder can be moved.
Every step writes into RUN (runs/20260924_A1_MAIN) and never modifies earlier run folders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# SITE 2 (Groningen-Drenthe) copy of A1 a1_common.py. Changed: RUN, G0L, MAN point into site2_20260930;
# load_scene_catalogue reads the site-2 manifest format (s2_01_manifests.py). Constants (classes, budgets,
# frozen HyP3 parameters, SEED used by the experiment RNG) are unchanged so both sites share one protocol.
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "MAIN"

G0L = ROOT / "runs" / "G0L_COMMON_GRID"
GRID_SPEC = G0L / "grid_spec.json"
MAN = ROOT / "manifests"
WRAP = ROOT / "runs" / "20260923_MAIN_PARAMS_WRAPPED" / "reports"
P0 = ROOT / "runs" / "20260923_P0_PILOT"

MAIN_PREFLIGHT = WRAP / "MAIN_SDK_PREFLIGHT.json"
SUBSET_B_CSV = WRAP / "CLOSURE_SUBSET_B.csv"

YEARS = (2025, 2024)          # 2025 = primary, 2024 = replication
TRACKS = (37, 88)
TARGET_CLASSES = ["PermGrass", "TempGrass", "Maize", "Potato", "Beet", "WinterCereal", "SpringCereal"]
BUDGETS = (1, 2, 3, 4, 6, 8, 9)       # number of disjoint 12-day pairs (M); scenes N = 2M
CREDIT_RTC = 15
CREDIT_INSAR = 15
SEED = 20260923

FROZEN_INSAR = dict(looks="10x2", include_wrapped_phase=True, phase_filter_parameter=0.6,
                    apply_water_mask=True, include_inc_map=True)
FROZEN_RTC = dict(radiometry="gamma0", resolution=20, scale="power")

D = {
    "submit": RUN / "submit",
    "downloads": RUN / "downloads",
    "aligned": RUN / "aligned_40m",
    "cube": RUN / "cube",
    "exp": RUN / "experiments",
    "stats": RUN / "stats",
    "fig": RUN / "figures",
    "reports": RUN / "reports",
    "logs": RUN / "logs",
}


def ensure_dirs() -> None:
    for p in D.values():
        p.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_logger(name: str) -> logging.Logger:
    ensure_dirs()
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(D["logs"] / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


def sha256_file(p: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


def read_json(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def update_status(step: str, state: str, **info) -> None:
    """Global status board that the agent (and the user) can read at any time."""
    p = D["reports"] / "PIPELINE_STATUS.json"
    st = read_json(p, {})
    st[step] = {"state": state, "utc": now(), **info}
    write_json(p, st)


# ---------------------------------------------------------------- scene / pair catalogue
_DT = re.compile(r"(\d{8}T\d{6})")


def dt_key(s: str) -> str:
    m = _DT.search(s)
    if not m:
        raise ValueError(f"no datetime in {s}")
    return m.group(1)


def load_scene_catalogue() -> dict:
    """key 'YYYYMMDDTHHMMSS' -> dict(year, track, date, platform, scene_id)."""
    cat = {}
    for y in (2024, 2025):
        with open(MAN / f"scene_manifest_{y}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                k = dt_key(r["scene_id"])
                cat[k] = dict(year=int(r["year"]), track=int(r["track"]), date=r["date"],
                              platform=r["platform"].replace("Sentinel-", "S"), scene_id=r["scene_id"])
    return cat


def match_scene(key: str, cat: dict, tol_s: int = 5):
    if key in cat:
        return cat[key]
    t0 = datetime.strptime(key, "%Y%m%dT%H%M%S")
    best = None
    for k, v in cat.items():
        if k[:8] != key[:8]:
            continue
        d = abs((datetime.strptime(k, "%Y%m%dT%H%M%S") - t0).total_seconds())
        if d <= tol_s and (best is None or d < best[0]):
            best = (d, v)
    return best[1] if best else None


def load_pair_catalogue() -> dict:
    """(year, track, date1, date2) -> pair row."""
    out = {}
    for y in (2024, 2025):
        with open(MAN / f"pair_manifest_{y}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[(int(r["year"]), int(r["track"]), r["date_1"], r["date_2"])] = r
    return out


def load_budget_design() -> dict:
    """(year, track, M) -> ordered list of pair_ids (nested, uniform design produced in G0)."""
    out = {}
    with open(MAN / "budget_design.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["branch"] != "MAIN_12D_AA":
                continue
            ids = sorted(r["pair_ids"].split(";"), key=lambda s: s.split("_")[5])
            out[(int(r["year"]), int(r["track"]), int(r["M"]))] = ids
    return out


def pair_dates(pair_id: str):
    # PAIR_2025_T37_Sen_Sen_20250308_20250320_12D
    p = pair_id.split("_")
    f = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return int(p[1]), int(p[2][1:]), f(p[5]), f(p[6]), int(p[7].rstrip("D"))


# ---------------------------------------------------------------- product names
RTC_RE = re.compile(r"^(S1[ABCD])_IW_(\d{8}T\d{6})_DV[PR]_RTC20_G_\w+$")
INS_RE = re.compile(r"^(S1[ABCD]{2})_(\d{8}T\d{6})_(\d{8}T\d{6})_VVP(\d{3})_INT40_G_\w+$")


def parse_product(stem: str):
    m = RTC_RE.match(stem)
    if m:
        return {"kind": "RTC", "key1": m.group(2)}
    m = INS_RE.match(stem)
    if m:
        return {"kind": "INSAR", "key1": m.group(2), "key2": m.group(3), "lag": int(m.group(4))}
    return None
