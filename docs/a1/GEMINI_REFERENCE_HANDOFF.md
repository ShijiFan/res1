# Gemini Antigravity: A1 reference handoff

Read [REFERENCE_PACK_20260923.md](REFERENCE_PACK_20260923.md) in full **before** executing the local taskbook at `E:\research\SAR\A1_observation_budget_20260923\GEMINI_ANTIGRAVITY_TASKBOOK.md`. Use the paper/data/code links as research inputs, not as substitute results. Source check date: 2026-09-23.

## Required first deliverable: one evidence matrix, no jobs

Create `E:\research\SAR\A1_observation_budget_20260923\runs\<run_id>\reports\PRIOR_ART_AND_RESOURCE_MATRIX.md` with one row per resource. Columns: `resource`, `DOI_or_official_URL`, `verified_year_version`, `question_already_answered`, `A1_claim_affected`, `data_year_region`, `feature_definition`, `metric_and_split`, `code_license`, `local_use`, `unverified_detail`. Use the seven publications and official data sources in the reference pack. Open the full Wang et al. paper/supplement and explicitly list every result that already covers coherence increment, cross-region transfer, feature cost, temporal subsampling or budget curves. If its budget analysis overlaps our proposed center, revise A1's novelty map before spending credits; do not invent a distinction.

## Then run G0-R, using references as concrete templates

1. Query ASF for 2024 and 2025 SLC **and** GRD. Preserve raw responses. Emit scene manifest with acquisition UTC, platform, relative orbit, direction, polarization, footprint, product type and source granule ID.
2. Enumerate same-track pairs with 12-day S1A/S1A as the cross-year main candidate and 2025 6-day A/C as a conditional branch. Verify burst/coverage, perpendicular baseline and pair endpoints. Save rejected pairs with reasons.
3. Load both definitive PDOK BRP releases separately. Record original CRS, `jaar`, crop-code distributions and stable per-year parcel IDs. Prepare an explicit annual code-to-class mapping; never import SandboxNL's old crop strings as if they were 2024/25 truth.
4. Make `job_manifest_dryrun.csv` with live HyP3 price, duplicate-job removal, credit total, remaining balance, account snapshot time and expected download size. Stop before submission. The local taskbook controls the paid-batch gate.
5. Define a test for potential OPERA RTC substitution: same scenes, overlapping footprints, units, grid, masks, parcel cohort and radiometric bridge to HyP3 RTC. Reject it from main O2/O3 if any of these do not match.

## Scientific protocol to lock before modeling

For each budget `M` pairs, use the **identical endpoints** for intensity-only O2 and intensity-plus-coherence O3; record `N_unique`, calendar span, first/last day and processing credits. Use parcel-level spatial folds, a 400 m geometric exclusion and saved out-of-fold predictions. Compute paired block-level ΔBA, uncertainty and `N*` exactly as prescribed in the local taskbook. Treat changes in temporal coverage, pair lag and parcel attrition as explicit design variables. Do not call a cross-paper number a baseline.

## Output format at each milestone

Report `command`, `config path`, `input hashes`, `manifest/prediction paths`, `quantitative result`, `failed assertions`, `credit cost`, `claim supported`, and `claim still untested`. If a reference link fails, identify the URL and time of access; continue unaffected local work. Never synthesize missing observations or silently change a split, label mapping, date schedule or evaluation metric.
