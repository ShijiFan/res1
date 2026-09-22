# Cross-Track Sentinel-1 Land-Cover Classification with Two-Acquisition Interferometric Coherence

[![Paper](https://img.shields.io/badge/IEEE_TGRS-Under_Review-blue.svg)](https://github.com/ShijiFan/res1)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Results-100%25_Verified-success.svg)](scripts/reproduce_all_tables.py)

Official open-source repository and reproducibility package for the paper:  
**"Cross-Track Sentinel-1 Land-Cover Classification with Two-Acquisition Interferometric Coherence"**  
Submitted to *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.

---

## 📖 Overview & Abstract

Interferometric coherence records temporal scattering stability that complements SAR backscatter intensity. Its contribution to cross-track land-cover classification at a strict two-acquisition observation budget ($N = 2$) is the focus of this study. 

We evaluate four Sentinel-1 tracks over the eastern Netherlands, spanning ascending and descending passes and $12.2^\circ$ in incidence angle. A single 12-day coherence measurement augments a 28-dimensional intensity representation derived from the same image pair. Spatial model selection and evaluation use a 400 m training exclusion ($k$-d tree guard), with three sampling seeds and twelve directed cross-track transfers.

```
       [ Sentinel-1 Single-Look Complex (SLC) Pair: t1, t2 ]
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[ SAR Intensity Backscatter ]             [ InSAR Coherence ]
  - Dual-pol: VV, VH (dB)                   - 12-day repeat-pass correlation
  - Cross-ratio RT = VH - VV                - Scalar coherence magnitude g12
  - Multi-scale spatial statistics          - Derived transforms (Sobel, text.)
  ───► Base Intensity: I8 (8-D)             ───► Scalar Coherence: I8 + g12 (9-D)
  ───► Full Baseline:  I28 (28-D)           ───► Full Fusion:      I28 + g12 (29-D)
                               │
                               ▼
        [ 400 m Spatial Guard + 3-Fold Spatial Cross-Validation ]
                               │
                               ▼
     [ 12 Directed Cross-Track Transfers (RF & SVC Across 4 Tracks) ]
  Delta BA = +6.30 pp [5.25, 8.19] (RF)  |  Delta BA = +6.99 pp [5.78, 9.39] (SVC)
  Built-up Recall: 67.2% -> 90.7% (RF)   |  Built-up Recall: 66.6% -> 93.0% (SVC)
```

### 🎯 Key Headline Findings

1. **Substantial Accuracy Gain**: Coherence increases five-class balanced accuracy by **$+6.30\pp$** ($95\%$ block-bootstrap interval: $[5.25, 8.19]$) for Random Forest and **$+6.99\pp$** ($[5.78, 9.39]$) for Support Vector Classifier.
2. **Built-Up Class Recovery**: The largest gain is concentrated in built-up land cover: its misclassification as forest falls from $17.5\%$ to $0.8\%$ for Random Forest, driving a **$+23.5\pp$** (RF) and **$+26.4\pp$** (SVC) recall jump.
3. **Compact Representation Sufficiency**: The compact 9-dimensional representation ($I_8 + g_{12}$) captures over $95\%$ of the total accuracy gain achieved by larger feature sets; complex spatial coherence derivatives ($F_{12}$) offer no statistically distinguishable improvement over the raw scalar $g_{12}$.
4. **Temporal Robustness**: Spring-trained models transferred to autumn acquisitions retain balanced-accuracy gains of **$+5.03\pp$** (RF) and **$+5.23\pp$** (SVC).

---

## 📂 Repository Structure

```
ShijiFan/res1/
├── README.md                      # Project documentation and reproduction guide
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment definition
├── .gitignore                     # Git ignore rules
│
├── configs/                       # Acquisition parameters and feature ladder specs
│   └── dataset_parameters.json    # Track geometry (T15, T37, T88, T139; theta: 32.8° - 45.0°)
│
├── splits/                        # Spatial partition definitions
│   └── split_manifest.json        # 40 spatial blocks (28 train, 12 test) & 400 m exclusion
│
├── src/                           # Core algorithmic pipeline
│   ├── feature_pipeline.py        # 5-level representation ladder feature extraction
│   ├── compute_coherence.py       # Normalized complex correlation calculation
│   ├── spatial_cv.py              # 3-fold spatial cross-validation with k-d tree buffer
│   ├── train_models.py            # Model training (RF 300 trees, RBF-SVC)
│   ├── cross_track_eval.py        # 12 directed cross-track transfers evaluation
│   └── bootstrap_inference.py     # 20,000 spatial block-bootstrap resamples & sign test
│
├── scripts/                       # High-level reproduction and figure generation scripts
│   ├── reproduce_all_tables.py    # 1-Click reproduction of Tables I, II, III, IV, V
│   ├── reproduce_all_figures.py   # 1-Click rendering of Figures 1 to 14
│   └── figures/                   # IEEE Transactions Times-style matplotlib figure scripts
│       ├── tgrs_style.py          # Times New Roman + STIX math style sheet
│       ├── fig2_partitions.py     # Study area and block partitioning (Fig. 2)
│       ├── fig4_perclass.py       # Per-class recall decomposition (Fig. 7)
│       └── fig6_coherence_distributions.py # Coherence distributions & stability (Fig. 8)
│
├── tables/                        # Precomputed benchmark results & audit CSVs
│   ├── T1_contrasts.csv           # Canonical headline contrasts and confidence intervals
│   ├── T1_main_results.csv        # Detailed per-seed cross-track metrics
│   ├── T1_per_class.csv           # Per-class confusion matrices and recall numbers
│   └── T1_transfer_matrix.csv     # Complete 4x4 transfer matrices
│
├── runs_R1/                       # Complete 120 evaluation run folders (400 m buffer protocol)
│   ├── RF_I8_t15_s17/             # Saved blocks_confusion.npy, predictions.parquet, parameters
│   └── ...                        # All 120 learner x representation x track x seed runs
│
└── manuscript/                    # Manuscript sources and compiled PDFs
    ├── main_tgrs.pdf              # Authoritative manuscript PDF (12 pages, hyperref blue links)
    ├── supplement.pdf             # Supplementary material PDF (5 pages)
    ├── main_tgrs.tex              # Main LaTeX document
    ├── supplement.tex             # Supplement LaTeX document
    └── references.bib             # 48 verified IEEE references (72.9% 2024-2026)
```

---

## ⚡ Quickstart: 1-Click Reproduction

You can reproduce all benchmark tables, balanced accuracy contrasts, overall accuracy gains, and class recall breakdowns in **less than 2 seconds** directly from the included evaluation tensors:

```bash
# 1. Clone the repository
git clone https://github.com/ShijiFan/res1.git
cd res1

# 2. Install dependencies (or use an existing conda environment)
pip install -r requirements.txt

# 3. Run the 1-click reproduction script
python scripts/reproduce_all_tables.py
```

### Expected Output

```
================================================================================
  IEEE TGRS BENCHMARK TABLE REPRODUCTION
  Cross-Track Sentinel-1 Land-Cover Classification (Two-Acquisition Coherence)
================================================================================
Successfully loaded all 120 evaluation confusion tensors from runs_R1/

--------------------------------------------------------------------------------
TABLE I: Cross-Track Balanced Accuracy (BA), Overall Accuracy (OA), and Transfer Gap (G)
--------------------------------------------------------------------------------
Learner Representation                       Cross-BA (%)     In-BA (%)      Gap G (pp)  
--------------------------------------------------------------------------------
RF     I_8 (8-D Base Intensity)              64.36 +/- 0.48    69.06 +/- 0.39    +4.70
RF     I_28 (28-D Full Intensity Baseline)   67.03 +/- 0.32    71.78 +/- 0.43    +4.76
RF     I_8 + g_12 (9-D Scalar Coherence)     72.97 +/- 0.34    77.32 +/- 0.58    +4.35
RF     F_12 (12-D Coherence Transforms)      72.69 +/- 0.59    77.16 +/- 0.70    +4.47
RF     I_28 + g_12 (29-D Full Fusion)        73.33 +/- 0.45    77.44 +/- 0.41    +4.12
SVC    I_8 (8-D Base Intensity)              63.99 +/- 0.43    68.90 +/- 0.43    +4.91
SVC    I_28 (28-D Full Intensity Baseline)   66.72 +/- 0.11    71.46 +/- 0.07    +4.74
SVC    I_8 + g_12 (9-D Scalar Coherence)     74.09 +/- 0.39    77.95 +/- 0.52    +3.86
SVC    F_12 (12-D Coherence Transforms)      73.89 +/- 0.23    77.78 +/- 0.46    +3.89
SVC    I_28 + g_12 (29-D Full Fusion)        73.71 +/- 0.35    78.15 +/- 0.42    +4.44

--------------------------------------------------------------------------------
KEY HEADLINE FINDINGS (Paired Coherence Increment over I_28 Intensity Baseline):
--------------------------------------------------------------------------------
[RF] Balanced Accuracy : 67.03% -> 73.33%  (Increment Delta = +6.30 pp)
[RF] Overall Accuracy  : 69.42% -> 71.37%  (Increment Delta = +1.95 pp)
[SVC] Balanced Accuracy : 66.72% -> 73.71%  (Increment Delta = +6.99 pp)
[SVC] Overall Accuracy  : 71.13% -> 72.78%  (Increment Delta = +1.64 pp)

--------------------------------------------------------------------------------
TABLE II: Class-Level Recall Breakdown (Pooled over 12 Transfers and 3 Seeds)
--------------------------------------------------------------------------------
Class Name   RF Baseline    RF Augmented   RF Delta   SVC Baseline   SVC Augmented  SVC Delta 
--------------------------------------------------------------------------------
Forest        76.15%         81.13%         +4.97 pp    78.30%         82.92%         +4.61 pp
Grassland     76.49%         76.97%         +0.48 pp    81.86%         81.82%         -0.04 pp
Cropland      45.95%         47.19%         +1.24 pp    38.52%         39.69%         +1.18 pp
Built-up      67.22%         90.71%        +23.48 pp    66.63%         93.01%        +26.38 pp
Water         69.30%         70.64%         +1.34 pp    68.30%         71.12%         +2.83 pp
--------------------------------------------------------------------------------
```

---

## 🎨 Figure Reproduction

To re-render all manuscript publication figures matching IEEE Transactions Times typography:

```bash
python scripts/reproduce_all_figures.py
```

Generated vector PDFs will be stored in `manuscript/figures/`.

---

## 📚 Citation

If you find this research, code, or benchmark datasets helpful, please cite our paper:

```bibtex
@article{Fan2026CrossTrackCoherence,
  author    = {Fan, Shiji and Albuquerque, Daniel and Pinho, Pedro and Shen, Ming},
  title     = {Cross-Track {Sentinel-1} Land-Cover Classification with Two-Acquisition Interferometric Coherence},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2026},
  note      = {Under Review},
  url       = {https://github.com/ShijiFan/res1}
}
```

---

## 📄 License & Acknowledgments

- **Code & Scripts**: Licensed under the [MIT License](LICENSE).
- **Data & Products**: Sentinel-1 SAR data is provided by the European Space Agency (ESA) via the Alaska Satellite Facility (ASF) HyP3 platform. Land cover references are from ESA WorldCover 10 m and EEA CORINE Land Cover 2018.
- **Funding**: This work is funded by national funds through FCT--Fundação para a Ciência e a Tecnologia, I.P., and EU funds under project/support UID/50008/2025--Instituto de Telecomunicações (DOI: [10.54499/UID/50008/2025](https://doi.org/10.54499/UID/50008/2025)).
