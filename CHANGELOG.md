# LAMMF-on-NSCLC — Change Log & Decision Record

*Everything we changed from the moment we first tried to boot the original
LAMMF repo to where we are now (feature normalization done; stratified split
next). Written as a checkpoint before the "make-it-right" work continues.*

---

## 0. Where we started

**Original repo:** `Radiotherapy-AI-Lab/LAMMF` — a multimodal survival model for
**head-and-neck cancer**. Architecture: a 3D CNN over CT + tumor mask, a GNN over
radiomics features, a text branch (PubMedBERT over clinical notes), and a
**Gaussian Naive Bayes late fusion** that combines the branch outputs into one
survival prediction.

**Our goal:** get it running, point it at the **NSCLC-Radiogenomics** lung-cancer
dataset from TCIA instead of head-and-neck, then improve it. This is *training our
own model on new data*, not reproducing the paper.

**The problem:** the published repo does **not** run as-is. It had roughly a dozen
bugs — undefined names, a broken dataset cache, calls into a text model that was
never importable, a loss function that wasn't defined, and a trainer that only
ever built one of the two models. Nothing ran end-to-end out of the box.

---

## 1. Booting the original repo — bugs found and fixed

These are the fixes that took the repo from "won't start" to "runs end-to-end on
synthetic data." Grouped by file.

### `data_utils.py` (the model's data loader — heavily reworked)
- **Class renamed** `HNC_OS_Dataset` → `NPC_OS_Dataset`, and every reference
  updated. Undefined-name errors were coming from a mismatch between the class
  definition and the code that instantiated it.
- **The dataset cache was broken.** PyTorch Geometric's `InMemoryDataset` expects
  `process()` to *collate* the list of `Data` objects and save them as a single
  `(data, slices)` tuple. The original saved the wrong thing, so rebuilding the
  cache produced a `NoneType` / invalid-type crash. Fixed to:
  ```python
  data, slices = self.collate(data_list)
  torch.save((data, slices), self.processed_paths[0])
  ```
  and in `__init__`:
  ```python
  self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
  ```
  (We spent real time understanding *why* `processed_paths[0]` is reused rather
  than incremented — it's a single cache file per split, overwritten by design,
  not a bug.)
- **Column contract pinned down.** The loader slices features as
  `columns[4:4+1302]`, i.e. it assumes the first four columns are
  `Patient_ID, Component_ID, ROI_Name, Mask_Path` and everything after is
  features. This is the contract `build_real_csvs.py` has to satisfy exactly.
- **Trimmed the `Data` object** to what the branches actually use: `x`,
  `edge_index`, `y`, `ct_path`, `mask_path`, `all_mask_path`. The unused
  `patient_id` / `text` / `node_names` fields were commented out because they were
  causing "invalid type" errors in the DataLoader.
- **Left in a debug print loop (~lines 63–68)** — flagged for deletion (backlog
  item D7), not yet removed.

### `trainer.py`
- **It only built ONE model.** The original trainer instantiated a single model,
  so there was no obvious way to train *both* the CNN and the GNN and then fuse.
  We split the run by branch so you can train each separately.
- **Rather than editing the file every run**, the plan settled on calling it by
  branch (`trainer.py gnn`, `trainer.py cnn`) so you don't hand-edit which model
  builds each time.
- **`WeightedBCE` loss was undefined.** The trainer referenced a loss that didn't
  exist. We restored a working BCE criterion (note: currently *unweighted* — see
  backlog T1, this is a known regression against the authors' intent).
- **Clarified the metric.** Output is a sigmoid probability; loss is BCE; **AUC**
  is the threshold-free ranking metric we actually trust. Per-branch "accuracy"
  uses `argmax` on a single-output logit, which is degenerate (always predicts
  class 0) — reporting only, not to be trusted.

### `Model_.py`
- **The text branch was dead weight.** `Model_TEXT` was referenced but not
  importable, and PubMedBERT/transformers weren't part of the runnable path. We
  removed the text branch from the run so fusion doesn't error waiting on text
  data that never arrives.
- **Noted (not yet fixed) that the CNN loads and preprocesses images *inside*
  `forward()`** every epoch — brutally slow and the wrong separation of concerns.
  Flagged as backlog M1.

### `late_fusion_GNB.py`
- Rewired to import from the local model, use `NPC_OS_Dataset`, drop the text
  branch, and load the two saved checkpoints (`best_model_gnn.pth`,
  `best_model_cnn.pth`).
- Model constructors pinned: `Model_GNN(177, 1)` and
  `Model_CNN(1, 128, (16,32,32))`.
- GaussianNB fusion runs end-to-end. Per-branch accuracy left with the documented
  degenerate `argmax` (trust AUC/fusion instead).

**Result of Section 1:** the repo runs end-to-end on synthetic data — CNN + GNN +
GNB fusion, no crashes.

---

## 2. Building the real-data pipeline (new files we wrote)

The original repo assumed data already in its bespoke CSV format. To feed it the
raw TCIA NSCLC download, we built a preprocessing pipeline from scratch. In run
order:

1. **`make_labels.py`** — clinical CSV → `labels_all.csv` (`Patient_ID, label`).
   Reads `encoding='latin-1'`, uses `Survival Status` + `Time to Death (days)`,
   with a `--cutoff_days` flag (730 = 2-year). Labeling rule:
   **1 = survived, 0 = died** (dead-past-cutoff → survived the window → 1;
   dead-within-cutoff → 0; alive → 1). ~211 patients; ~175 survived / 36 died at
   2yr. Known caveat: "alive" patients aren't follow-up-checked (censoring — see
   deferred).

2. **`make_cohort.py`** — *diagnostic only.* Hits the TCIA API to list which
   patients actually have imaging, intersects with labels to estimate usable N.
   Not required for correctness.

3. **`convert_dicom.py`** — DICOM → NIfTI. Walks each patient's
   Patient→Study→Series→slice hierarchy, classifies series by Modality tag
   (CT/PT/SEG), reads the CT with SimpleITK's `ImageSeriesReader`, and converts the
   DICOM-SEG mask with `pydicom_seg` (merges segments, resamples to the CT grid
   with nearest-neighbor). Outputs `ct/<id>.nii.gz` + `masks/<id>.nii.gz`. Flags
   `--limit`, `--require_seg`. **Picks the LARGEST CT series** per patient — a
   crude heuristic (backlog D3: should use the CT the SEG explicitly references).

4. **`extract_features_mirp.py`** — CT + mask → `radiomics_features.csv`.
   **177 IBSI-compliant features via MIRP**, 2 mm isotropic resample
   (`new_spacing=2.0`), `fixed_bin_number=32`. Drops MIRP's metadata columns.
   Robust to empty masks (skips `None`/empty results gracefully).

5. **`build_real_csvs.py`** — the assembler. Reads `labels_all.csv` +
   `radiomics_features.csv` + the `ct/` and `masks/` folders, **intersects** to the
   cohort that has *all* of CT ∩ mask ∩ label ∩ features, splits **70/15/15**, and
   writes every file `data_utils.py` expects:
   `normalized_data_csv/normalized_data_{split}.csv`,
   `train_data_csv/{split}_set.csv`, `csv_data/file_paths_CT.csv`,
   `all_clin_data.csv`, and `Text_data/*.csv` placeholders. Column order is pinned
   to the loader's contract. (Has an `N_DUMMY_FEAT = 1302` fallback that emits
   random features if no radiomics file is present — used only for smoke tests.)

6. **`make_fake_data.py`** — a smoke-test generator. Creates 20 fake patients with
   `ct/` + `masks/` NIfTI volumes (noise + an intensity-correlated tumor cube) and
   a matching `labels_all.csv`, so the whole
   extract → build → train → fuse chain can be verified on **any machine with no
   DICOM download**. This is what we use to confirm a fresh environment works.

**Result of Section 2:** full pipeline runs end-to-end on **real** NSCLC data with
**real** IBSI radiomics features.

---

## 3. Key technical decisions (the "why", not just the "what")

- **PyRadiomics → MIRP.** PyRadiomics would not install on Python 3.13 / Apple
  Silicon (missing versioneer, numpy build failures, no compiler). MIRP is
  pip-installable, IBSI-compliant, and validated — so we switched. It produces 177
  features rather than PyRadiomics' larger default set; that's a deliberate,
  acceptable trade (backlog D6 revisits the feature set with the advisor).

- **Text branch dropped.** NSCLC-Radiogenomics doesn't come with the clinical-note
  text the original text branch expected, and the branch wasn't importable. We run
  CNN + GNN + fusion only. Re-adding text is out of scope for now.

- **GNN kept, even though the graph is trivial.** With one ROI per patient the
  "graph" is a single node, so message-passing does nothing — the GNN is
  effectively an MLP right now. We're *keeping* it on purpose: adding more nodes
  per patient is the whole point of the project going forward, so the scaffolding
  stays.

- **Binary survival framing.** Survive-past-cutoff = 1. Sigmoid output =
  probability of survival. AUC is the metric we trust because it's
  threshold-free and robust to the class imbalance.

- **Numbers are not yet meaningful — and we know it.** Only **39** usable patients
  (see below). With ~1 death in a 7-patient test set, per-branch AUC swings between
  0 and 1 across runs. This is a *"the machinery works"* milestone, not a
  performance result.

---

## 4. Current cohort & last run

- Downloaded + converted **40** R01 patients that have segmentations (only R01
  patients have SEG masks; AMC patients have none). **R01-022 has an empty mask**
  and was dropped → **39 usable patients**.
- Split: **27 train / 5 val / 7 test**.
- Fusion runs clean end-to-end. Reported numbers are meaningless at this N — trust
  only "it ran."

---

## 5. Handoff & environment work

Done to get the project onto the advisor's Windows machine and, later, a GPU
cluster:

- **GitHub repo + `.gitignore`.** Configured so `git add .` commits **only** code
  (`*.py`) and docs (`*.md`) — all imaging data, generated CSVs, caches, weights,
  and environment folders are ignored. (Learned the hard way that `.gitignore`
  comments must be on their **own line**, never trailing a pattern.)
- **`requirements.txt`** with critical pins: `pydicom==2.4.4` (3.x breaks
  `pydicom-seg`), `numpy<2` (2.x breaks several deps), plus a note to install
  **torch first** from pytorch.org before the rest.
- **`SETUP.md`** — step-by-step conda setup (Python **3.12**, because 3.13 has no
  `numpy<2` wheels and needs a C compiler nobody has), torch install, verify block.
- **Windows reality check:** `nvidia-smi` wasn't recognized →
  `Get-CimInstance Win32_VideoController` showed **Intel graphics only** → that
  machine is **CPU-only**. Real full-resolution training will need the **GPU
  cluster** (deferred).
- **PyCharm** interpreter finally pointed at the working conda env after a detour
  where it created an empty env; packages reinstalled, imports resolve.

---

## 6. Most recent change — feature normalization (backlog D1, ✅ done)

The 177 MIRP features come out on **wildly different scales**, and were being fed
raw into the GNN. We added **z-score standardization** (`StandardScaler`):

- **Fit on the TRAIN split only**, then `transform` val and test with those same
  train statistics — so no information leaks from val/test into the scaling.
- **Only the 177 feature columns** are scaled; `Patient_ID`, `Component_ID`,
  `ROI_Name`, `Mask_Path` are left untouched.
- Zero-variance features (constant across the tiny cohort) are handled by
  `StandardScaler` automatically (scale set to 1 → column becomes zeros, not NaN).
- **Not yet done:** persisting the fitted scaler (`joblib.dump`) so the exact same
  train statistics can be reapplied to genuinely new patients on the cluster.
  Optional now, important later.

---

## 7. Stratified split (backlog D2, ✅ done)

The old split did `random.shuffle(cohort)` and sliced blindly. At ~17% deaths that
routinely handed val/test **zero** minority patients, making evaluation meaningless
and training unstable. Replaced it with a **two-step stratified split** using
sklearn's `train_test_split`:

- Build a label list aligned to `cohort`: `y = [label_map[pid] for pid in cohort]`.
  (`label_map` is the `{pid: label}` dict; the split needs classes as a parallel
  list so it can balance them.)
- **Step 1:** peel off 30% as a temp pool, stratified on `y`.
- **Step 2:** split that 30% pool **50/50** into val and test, stratified on the
  pool's own labels. Net result: **70/15/15**, class ratio preserved in *every*
  split.
- The patient-level split is shared by both branches — a patient's image (CNN) and
  features (GNN) always land in the same split — which is what keeps late fusion
  valid (no patient is train for one branch and test for the other).

Honest caveat unchanged: at N≈39 with ~6–7 deaths, stratification guarantees the
minority isn't *zero* and keeps proportions sane, but can't manufacture
statistical power. Cross-validation + scale-up remain the real fix.

---

## 8. Random seeds (backlog T3, ✅ done)

The trainer had no seeds, so every run started from different random weights and a
different shuffle — meaning you couldn't tell whether a change helped or you just
got lucky. Added at the top of `trainer.py`, **before the model is built**:

```python
np.random.seed(42)
torch.manual_seed(42)
```

`torch.manual_seed` is the one that matters here — it covers weight initialization,
the DataLoader shuffle, and dropout (all PyTorch). `np.random.seed` is insurance;
Python's `random` isn't seeded because the trainer doesn't use it. cuDNN
determinism flags are deferred — they only matter on GPU. This makes every
subsequent change *measurable*: same seed → same run, so a moved metric is the
change, not luck.

---

## 9. Weighted BCE loss (backlog T1, ✅ done)

With ~17% deaths and unweighted BCE, the model minimized loss by predicting
"survived" for everyone — learning nothing, AUC ~0.5. Fixed with **per-sample
weighted BCE**, kept deliberately minimal (two edits, all in `trainer.py`):

- Two inverse-frequency class weights computed once from the labels:
  `W_DIED ≈ 2.9`, `W_SURVIVED ≈ 0.6` (a death now counts ~5× a survivor).
- Inside the training loop, `torch.where(label == 0, W_DIED, W_SURVIVED)` stamps
  those weights onto each batch, and `F.binary_cross_entropy(pred, label,
  weight=weights)` applies them.

**Design choice:** we kept the model's `sigmoid` + `BCELoss` route and weighted via
a per-sample tensor, rather than switching to `BCEWithLogitsLoss(pos_weight=...)`
(which would have meant removing the sigmoid from the model and applying it in the
fusion code instead). The per-sample route keeps the whole change in one file.
**Validation loss stays unweighted** — early stopping should reflect true
performance, not the training-time reweighting.

---

## 10. Where we are & what's next

**Status:** pipeline runs end-to-end on real data. Three make-it-right fixes are
in: features **normalized** (D1), split **stratified** (D2), seeds set (T3), and
loss **weighted** (T1). The model should no longer collapse to "survived."

**Immediate next:** **M1 — move CNN preprocessing out of `forward()`.** The CNN
currently loads and preprocesses image volumes *inside* `forward()`, every epoch —
brutally slow and it blocks augmentation. Moving it into the Dataset (cache once)
makes CNN iteration fast and sets up M2/M3/M6.

**Then, in rough order:** `M2/M3/M6` regularization + normalization + augmentation
(attack overfitting) → `D5/M4` the GNN-vs-MLP design decision → scale-up on GPU.
Full detail lives in `ISSUES_BACKLOG.md`.

**Deferred (advisor/compute-gated):** censoring in labels, 2yr-vs-5yr cutoff,
head/neck→lung framing, GPU for full-res + full cohort, cross-validation.
