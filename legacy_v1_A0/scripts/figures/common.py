import sys, csv, json
from pathlib import Path
import numpy as np

BASE = Path(r"E:\research\SAR\tgrs_final_campaign_20260915")
T_DIR = BASE / "tables"
R_DIR = BASE / "runs_R1"
S_DIR = BASE / "splits"
OUTDIR = Path(__file__).resolve().parents[1] / 'checked_manuscript/figures'
OUTDIR_ROOT = Path(__file__).resolve().parents[1] / 'figure_previews'

MAIN = {(r['learner'], r['representation']): r for r in csv.DictReader(open(T_DIR / 'T1_main_results.csv', encoding='utf-8'))}
def M(L, R, k): 
    return float(MAIN[(L, R)][k])

PC = list(csv.DictReader(open(T_DIR / 'T1_per_class.csv', encoding='utf-8')))

CON = {(r['contrast_id'], r['learner'], r['metric']): r for r in csv.DictReader(open(T_DIR / 'T1_contrasts.csv', encoding='utf-8'))}
def C(cid, L, metric):
    r = CON[(cid, L, metric)]
    return float(r['point_estimate']), float(r['ci_lo']), float(r['ci_hi'])

TM = list(csv.DictReader(open(T_DIR / 'T1_transfer_matrix.csv', encoding='utf-8')))

# Okabe-Ito assignments for the representation ladder
REP_ORDER = ['I8', 'I28', 'I8_G9', 'F12', 'I28_G29']
REP_LABEL = {
    'I8': '$I_8$\n(8-D)',
    'I28': '$I_{28}$\n(28-D)',
    'I8_G9': '$I_8{+}g_{12}$\n(9-D)',
    'F12': '$F_{12}$\n(12-D)',
    'I28_G29': '$I_{28}{+}g_{12}$\n(29-D)'
}
REP_COL = {
    'I8': '#8c8c8c',        # Gray
    'I28': '#D55E00',       # Vermillion
    'I8_G9': '#0072B2',     # Blue
    'F12': '#56B4E9',       # Sky Blue
    'I28_G29': '#009E73'    # Green
}
LEARNERS = [('RF', 'Random Forest (RF)'), ('SVC', 'Support Vector Classifier (SVC)')]
