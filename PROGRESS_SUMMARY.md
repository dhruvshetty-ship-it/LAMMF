# LAMMF on NSCLC-Radiogenomics — Progress Summary

*Living snapshot. Source of truth for the project — keep this current.*
*Last updated: after normalization, stratified split, seeds and weighted loss.*

## The task
Get the LAMMF multimodal survival model (3D CNN + GNN + late fusion) working,
then run it on the NSCLC-Radiogenomics dataset, then improve it.

## STATUS: full pipeline runs end-to-end on REAL data
- The published LAMMF repo does NOT run as-is (~a dozen bugs). We have a working,
  debugged version.
- CNN + GNN + Gaussian-Naive-Bayes fusion all run end-to-end on real NSCLC data,
  with real IBSI radiomics features.
- **Numbers are not yet meaningful** (only 39 usable patients, tiny/imbalanced
  test set) — this is a "the machinery works" milestone, not a performance one.

## The data pipeline (built + tested, in order)
Raw TCIA download (DICOM CT + DICOM-SEG masks + clinical CSV) ->
1. **make_labels.py** — clinical CSV -> `labels_all.csv` (binary survival at a
   cutoff; 2yr = 730 days). 211 patients; ~175 survived / 36 died at 2yr.
2. **make_cohort.py** — diagnostic only: intersect labels with TCIA imaging list
   (from API, no download) to estimate usable N. NOT required for correctness.
3. **convert_dicom.py** — DICOM -> NIfTI. `ct/<id>.nii.gz` + `masks/<id>.nii.gz`.
   Flags: `--limit`, `--require_seg`. Picks LARGEST CT series per patient (crude).
4. **extract_features_mirp.py** — CT+mask -> `radiomics_features.csv` (177 IBSI
   features via **MIRP**, resampled to 2mm). NOTE: PyRadiomics was abandoned
   (won't install on Python 3.13 / Apple Silicon); MIRP is the validated,
   pip-installable replacement.
5. **build_real_csvs.py** — assembles, stratified-splits (70/15/15) and
   z-score-normalizes (fit on train) into the exact CSVs
   `data_utils.py` reads: normalized_data_*, *_set, file_paths_CT, plus clin/text
   placeholders. Intersects label ∩ CT ∩ mask ∩ features.
6. **data_utils.py** (the model's loader, heavily modified) — reads the CSVs,
   builds PyG `Data` objects (x=features, edge_index, y=label, ct/mask paths).

## Current cohort
- Downloaded + converted **40** R01 patients (with masks). Only R01 patients have
  segmentations; AMC patients have none. 1 empty mask (R01-022) dropped -> **39**.
- Split (stratified): 27 train / 6 val / 6 test.

## Last real-data fusion result (before the fixes; MEANINGLESS numbers)
Per-branch AUCs swung wildly (CNN test 0.0-1.0 across runs) because the test set
had 1 death in 7 patients. Not yet re-run with the four fixes in. Per-branch "accuracy" is degenerate (argmax on a
single-output = always predicts class 0). Trust nothing here except "it ran."

## IMPROVEMENT AUDIT (get-it-working done; now make-it-right)

### DATA-side issues (ranked)
1. ~~**Radiomics features not normalized**~~ — DONE: z-score (StandardScaler), fit
   on train only, transform val/test. (Scaler not yet persisted via joblib.)
2. ~~**Split is random, not stratified**~~ — DONE: two-step stratified split
   (70/15/15), class ratio preserved in every split.
3. **CT selection = "biggest series"** — should use the CT the SEG *references*
   (DICOM-SEG stores it) to guarantee mask/CT alignment. Correctness risk at scale.
4. **Empty masks** (e.g. R01-022) — understand why (label value? geometry?) before
   scaling to 144.
5. **Trivial GNN graph** — one ROI/patient -> single-node graph -> GNN is really an
   MLP. Design decision: keep as MLP, drop for a tabular model (often better for
   single-tumor radiomics), or build multi-node graphs. Biggest conceptual call.
6. Cosmetic: delete debug print loop in data_utils.py (lines ~63-68).

### MODEL/TRAINING-side issues (ranked)
0. ~~**No random seeds**~~ — DONE: np + torch seeded in trainer (reproducible runs).
1. ~~**Class imbalance unhandled**~~ — DONE: per-sample weighted BCE (deaths ~5x
   survivors) via F.binary_cross_entropy. Model no longer collapses to "survived".
2. **CNN crop is (16,32,32)** — smoke-test size, throws away the scan. Bump toward
   (64,224,224) for real training (needs GPU).
3. No data augmentation (small N -> overfitting).
4. CNN does disk I/O in forward() every epoch (slow; refactor to cache).
5. Single (stratified) split, no cross-validation (high-variance estimates at N=39).

## Deferred (from DEFERRED_ISSUES.md)
Censoring in labels (148 "alive" not follow-up-checked), cutoff choice (2yr/5yr),
head/neck-vs-lung framing, GPU needed for full-res + full cohort.

## Next steps
- Re-run GNN + CNN + fusion with the four fixes in (delete `dataset/` first) and
  check whether predictions stop collapsing to "survived."
- Then M1: move CNN image preprocessing out of `forward()` into the Dataset.
- Full ordered list in ISSUES_BACKLOG.md.

## Tooling (all in LAMMF_IN_PROGRESS, tested)
make_labels.py, make_cohort.py, convert_dicom.py, extract_features_mirp.py,
build_real_csvs.py. Model: Model_.py, data_utils.py, trainer.py, late_fusion_GNB.py.
