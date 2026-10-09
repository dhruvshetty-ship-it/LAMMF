# LAMMF on NSCLC-Radiogenomics

Survival prediction for **non-small cell lung cancer** using a multimodal model:
a **3D CNN** on CT + tumor mask, a **GNN** on radiomics features, and
**Gaussian Naive Bayes late fusion** of the two.

This is an adaptation of the LAMMF framework
([Radiotherapy-AI-Lab/LAMMF](https://github.com/Radiotherapy-AI-Lab/LAMMF)),
originally built for head-and-neck cancer. The published code did not run as-is;
this repo is a debugged version retargeted at the
[NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/)
collection from TCIA. We are training our own model on new data, not reproducing
the paper's results.

> **Status:** the full pipeline runs end-to-end on real data. Results are **not yet
> meaningful**: only ~39 patients currently have usable CT + segmentation, so the
> test set is tiny. See `PROGRESS_SUMMARY.md` and `ISSUES_BACKLOG.md`.

---

## What changed from the original LAMMF

| | Original LAMMF | This repo |
|---|---|---|
| Cancer / data | Head & neck, 6 TCIA centers | Lung (NSCLC-Radiogenomics) |
| Text branch (PubMedBERT) | Used | **Removed** (no clinical-note text in this dataset) |
| Radiomics | 1302 features | **177 IBSI features via MIRP** (PyRadiomics won't install on Python 3.13 / Apple Silicon) |
| Label | Overall survival | Binary: survived past a cutoff (default 2 years). `1 = survived`, `0 = died` |
| Training | One model, hand-edited per run | `trainer.py gnn` / `trainer.py cnn` |

Improvements made since getting it running: z-score feature normalization
(fit on train only), stratified train/val/test split, fixed random seeds, and
class-weighted BCE loss. Full history in `CHANGELOG.md`.

---

## Setup

See **`SETUP.md`**. Short version: Python 3.12, a virtual environment, install
PyTorch first, then `pip install -r requirements.txt`.

---

## Running the pipeline

Run everything from the repo folder with your environment active
(`source .venv/bin/activate`). Raw data, generated CSVs, caches and model weights
are all gitignored, so each machine builds them locally.

**1. Labels** — clinical CSV from TCIA → `labels_all.csv`
```
python make_labels.py <clinical.csv> --cutoff_days 730
```

**2. Images** — DICOM download → `ct/<id>.nii.gz` and `masks/<id>.nii.gz`
```
python convert_dicom.py "/path/to/NSCLC download root" --out . --require_seg
```

**3. Radiomics** — CT + mask → `radiomics_features.csv` (177 features)
```
python extract_features_mirp.py
```

**4. Assemble + split + normalize** → all CSVs the model reads
```
python build_real_csvs.py labels_all.csv
```
Then **delete the `dataset/` folder** if it exists, so the loader rebuilds its
cache from the new CSVs.

**5. Train each branch**
```
python trainer.py gnn
python trainer.py cnn
```
Best weights are saved to `best/best_model_gnn.pth` and `best/best_model_cnn.pth`.

**6. Fuse**
```
python late_fusion_GNB.py
```
Trust the **AUC** numbers, not accuracy, at this sample size.

### Smoke test without downloading data
`make_fake_data.py` creates 20 fake patients (`ct/`, `masks/`, `labels_all.csv`),
replacing steps 1–2. Then run steps 3–6 to confirm a fresh environment works.

---

## Files

| File | Purpose |
|---|---|
| `make_labels.py` | Clinical CSV → binary survival labels |
| `make_cohort.py` | Diagnostic only: estimates how many patients have imaging |
| `convert_dicom.py` | DICOM CT + DICOM-SEG → NIfTI |
| `extract_features_mirp.py` | IBSI radiomics with MIRP |
| `build_real_csvs.py` | Builds the cohort, stratified split, normalization, model CSVs |
| `make_fake_data.py` | Fake dataset for smoke-testing |
| `data_utils.py` | PyTorch Geometric dataset (`NPC_OS_Dataset`) |
| `Model_.py` | Model definitions (`Model_CNN`, `Model_GNN`) |
| `trainer.py` | Trains one branch (`gnn` or `cnn`) |
| `late_fusion_GNB.py` | Fuses the two branches with Gaussian Naive Bayes |

## Project docs

- `SETUP.md` — environment setup
- `PROGRESS_SUMMARY.md` — current state of the project
- `ISSUES_BACKLOG.md` — known issues, ranked, with what's done
- `CHANGELOG.md` — everything changed from the original repo, and why

## Known limitations

Small cohort (~39 patients), so all metrics are high-variance. The GNN currently
sees one node per patient, so it behaves like an MLP. The CNN uses a small crop
(16×32×32) and preprocesses images inside `forward()`. Labels don't yet account
for censoring. Full-resolution training needs a GPU. Details in
`ISSUES_BACKLOG.md`.
