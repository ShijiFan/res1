# When Does Repeat-Pass Coherence Add Information?

Code, frozen protocols, result tables and predictions for the manuscript

> S. Fan, D. Albuquerque, P. Pinho, and M. Shen, "When Does Repeat-Pass Coherence Add Information?
> Sentinel-1 Land-Cover and Crop Mapping Across Acquisition Budgets, Class Definitions, and Spatial
> Context," manuscript prepared for *IEEE Transactions on Geoscience and Remote Sensing*.

The paper measures the marginal value of Sentinel-1 12-day VV coherence over backscatter from the same
acquisitions at two Dutch sites:

- **Study 1** (two acquisitions, five land-cover classes, four tracks T15/T37/T88/T139, May 2024; October
  2024 temporal replication): pixel classifiers (RF, SVC), a spatial CNN on 9×9 patches, and a check
  against the Dutch building register (BAG).
- **Study 2** (acquisition budget M = 1…9 pairs, seven BRP crop classes, tracks T37/T88, 2025 and 2024):
  HGB/RF/LR budget curves, equal-cost comparisons, polarization, and a TempCNN temporal baseline.

Site A (Gelderland–Overijssel) is the development site. Site B (Groningen–Drenthe) is an independent
replication run under a protocol frozen before any site-B SAR data were processed.

## Repository layout

| Path | Contents |
|---|---|
| `site_A/study1/` | Study 1 at site A: `PROTOCOL.md` (γ0 rerun protocol, frozen before results), `RESULTS.md`, `scripts/s1…s16`, `results/` (tables, BCa intervals, CNN checks, BAG check, selected configs, splits), `predictions/` (per-pixel predictions, per-block confusion matrices) |
| `site_A/study2/` | Study 2 at site A: `prereg_and_qc/` (pre-registration, E0 QC, R2 robustness check), `scripts/` (`a1_s1…a1_s9`, `a1_r2_*`, G0 audit/manifest scripts), `manifests/`, `grid_labels_splits/` (grid, parcel tables, spatial folds), `results/stats*`, `predictions/` (experiment registry and per-parcel predictions) |
| `site_B/` | Replication: `G0_and_prereg/` (feasibility check, pre-registration), `scripts/` (`s2_01…s2_20`, run scripts), `manifests/`, `study1/`, `study2/` (same structure as site A), `RESULTS_SITE2.md`, `RESULTS_SITE2_STUDY2_CRITERIA.json` |
| `temporal_dl/` | TempCNN baseline for Study 2 at both sites: `PROTOCOL.md`, `tdl_run.py`, `tdl_stats.py`, per-site registries, predictions, paired cells and judgement |
| `figures/` | `make_figures.py`, which builds every manuscript figure from the tables above |
| `src/` | Shared Study 1 training/evaluation code (`run_R1_buffered_cv_retrain.py`), imported unchanged by the site-A v7 and site-B scripts |
| `legacy_v1_A0/` | **Superseded.** The earlier single-site A0 package (σ0/γ0-mixed intensity, dB-domain resampling). Kept for provenance only; all numbers in the current manuscript come from the folders above |
| `tools/assemble_release.py` | Script that copied this release from the working folders |

Protocols, result notes and reports were written in Chinese during the project; tables, code and
comments are in English.

## Frozen protocols

Each protocol was hashed (SHA-256, LF line endings) before the results it governs were computed.

| Protocol | SHA-256 |
|---|---|
| `site_A/study2/prereg_and_qc/A1_PREREG.md` | `629ef45fde055988650754b5e8f79f89a3b811b762f5ad1eb1e1971b89d2c2eb` |
| `site_B/study2/prereg_and_qc/SITE2_PREREG.md` (frozen copy) | `58970ccc7c7732eefbc9244a28b4809b3f3beb05bc9c13bd5e6803627bde93e5` |
| `temporal_dl/PROTOCOL.md` | `de5a7cd5ee23915f28d8282683ff561f8eb1dec5729d96129864e0b788cc6add` |

`site_B/G0_and_prereg/PREREG_SITE2.md` is the same document with deviation note D1 appended on
2026-10-01, before any site-B Study 1 result existed; it therefore no longer matches the frozen hash.
Verify with `sha256sum` (or `certutil -hashfile <file> SHA256` on Windows).

## Data sources

No SAR rasters, data cubes or trained models are included (they total several hundred GB and are
rebuildable). All inputs are public:

- **Sentinel-1** IW SLC scenes (ESA Copernicus), processed by **ASF HyP3** (GAMMA): `RTC_GAMMA`
  (γ0 power, 20 m; 10 m for the site-A autumn window; Copernicus GLO-30 DEM) and `INSAR_GAMMA`
  (10×2 looks, Goldstein–Werner α = 0.6, water mask). Scene and pair lists are in the `manifests/`
  folders and `results/cube_index/PRODUCT_INDEX.csv`; job parameters are in the submission scripts.
- **ESA WorldCover 2021** and **CORINE Land Cover 2018** (Study 1 labels).
- **BRP Gewaspercelen**, definitive releases 2024 and 2025 (PDOK; Study 2 labels).
- **BAG** building register (PDOK OGC API), downloaded by `s9_download_bag.py` / `s2_17_download_bag.py`.

HyP3 submission needs NASA Earthdata credentials, read from `~/.netrc` or the environment variables
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`. No credentials are stored in this repository.

## Reproducing

```bash
conda env create -f environment.yml   # env "sar-tgrs", Python 3.10
conda activate sar-tgrs
```

The scripts were run from working folders on the authors' machine and contain those absolute paths
(`E:\research\SAR\...`); set them to your own layout before running. The mapping is:

| Release folder | Original working folder |
|---|---|
| `site_A/study1` | `SAR/A0_gamma0_rerun_20260929` (outputs under `ws/`) |
| `site_A/study2` | `SAR/A1_observation_budget_20260923` (main run `runs/20260924_A1_MAIN`) |
| `site_B` | `SAR/site2_20260930` and `SAR/site2_G0_20260930` |
| `temporal_dl` | `SAR/temporal_dl_20261001` |
| `src` | `SAR/res1_clean/src` |

Order of execution: site A Study 1 `run_pipeline.sh` then `run_checks.sh`; site A Study 2
`a1_s1` … `a1_s9`, then `a1_r2_*`; site B `run_site2_s2.sh` (Study 2) and `run_site2_s1.sh` (Study 1),
then `s2_20_judge_study2.py`; `temporal_dl/run_all.sh`; finally `figures/make_figures.py`.

The statistical summaries (bootstrap intervals, paired contrasts, replication criteria) can be
recomputed from the included predictions and confusion matrices without any SAR data, e.g.
`site_A/study1/scripts/s5_t1_v7.py`, `s11_contrasts.py`, `s16_autumn_blocks.py`,
`site_A/study2/scripts/a1_s7_stats.py`, `a1_r2_stats.py`, `site_B/scripts/s2_15_bootstrap.py`, and
`temporal_dl/tdl_stats.py`.

## License

MIT (see `LICENSE`). Third-party data keep their own licenses (Copernicus, ESA WorldCover CC BY 4.0,
CORINE, PDOK/BRP and BAG open data).
