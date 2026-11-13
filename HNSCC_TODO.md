# HNSCC dataset — to-do log

*Started Oct 9, 2026; refreshed Nov 13, 2026. Data lives only on the advisor's
Windows PC at `V:\QIXR\TCIA_H&N\HNSCC`. Code goes Mac
(`~/Python/ai_lab/LAMMF_IN_PROGRESS`) → GitHub → Windows.*

## Where things stand

- **Inventory done** (`inventory_data.py`): 65 patient folders, not the expected
  ~130. 61 have an RT Simulation study, 60 have an RTSTRUCT, and every RTSTRUCT
  references a CT in the same study. 2 patients have two CT series. The clinical
  sheet (`Table 1_update`) covers 215 patients, and all 65 imaged patients match
  it by `TCIA code`.
- **CT conversion tested** on 3 patients: 2 converted, 1 skipped (PET only,
  expected). One scan showed **uneven slice spacing** (a gap ~7 mm off). The
  converter now has a spacing check (regenerated Nov 13; push it).
- **Labels script run** (`make_labels_hnscc.py`) with a temporary simplified rule
  (backlog L1). At a 2-year cutoff: 56 labeled, only **4 deaths**. Default changed
  to a **5-year** cutoff; re-run pending.
- **Masks not started.** Waiting on the GTV choice (backlog H1).

---

## Next up

### 1. Mac housekeeping
- [ ] The project folder moved, so rebuild the venv (`conda deactivate` first):
      `rm -rf .venv` → `python -m venv .venv` → `source .venv/bin/activate` →
      `python -m pip install --upgrade pip` → `python -m pip install torch` →
      `python -m pip install -r requirements.txt`
- [ ] Copy in the regenerated files (`convert_ct_hnscc.py`, `HNSCC_TODO.md`,
      `ISSUES_BACKLOG.md`, `CHANGELOG.md`, `PROGRESS_SUMMARY.md`), then
      `git add .` → `git commit -m "..."` → `git push`.

### 2. Run the scripts (Windows)
- [ ] `git pull`
- [ ] Check that `ct\` only holds `HNSCC-...` files. If old fake smoke-test files
      (`R01-...`) or old `masks\`, `dataset\` or `labels_all.csv` are there, delete them.
- [ ] Labels:
      `python make_labels_hnscc.py "<clinical.xlsx>" --patients_root "V:\QIXR\TCIA_H&N\HNSCC"`
  - [ ] Check each cause-of-death category landed in the right bucket (kept vs removed).
        Watch for anything like "other cancer".
  - [ ] Record the final count: how many labeled, how many died vs survived.
- [ ] CTs: `python convert_ct_hnscc.py "V:\QIXR\TCIA_H&N\HNSCC"`
  - [ ] Expect ~61 converted, 4 skipped, no errors.
  - [ ] Record the **slice spacing** summary: how many scans are UNEVEN, and how bad.

### 3. Decide on uneven scans (backlog H4)
- [ ] Few uneven scans → drop them for now.
- [ ] Many → resample each one onto an evenly spaced grid using the true slice
      positions (needs a small addition to `convert_ct_hnscc.py`).

### 4. Message your advisor
- [ ] Is 65 folders the full dataset, or are ~65 more patients somewhere else?
- [ ] Primary GTV only (agreed). For patients with both, pre-chemo or post-chemo GTV?
- [ ] 8 patients have no GTV-named structure. Exclude them?
- [ ] OK to use the simplified label rule for now (cancer deaths only, other deaths
      removed, 5-year cutoff)? Should "Other Cancer" deaths (2 patients) count?
- [ ] Mention any uneven-slice scans found.

---

## Before training can run

### A. Masks — the main blocker
- [ ] Get the GTV answers from step 4.
- [ ] Settle the uneven-scan question first: the mask must sit on exactly the same
      grid as the saved CT.
- [ ] Check `rt-utils` installs alongside `pydicom==2.4.4`. If not, use `platipy` or
      fill the contours ourselves with `scikit-image`. Add the winner to `requirements.txt`.
- [ ] Write `convert_rtstruct_hnscc.py`:
  - [ ] Choose the primary GTV per patient by name. Clear primaries: `gtv primary`,
        `gtv bot`, `gtv tonsil`. Exclude node structures and `ptv70_nogtv`.
  - [ ] For plain `gtv` (34 patients), decide if it's primary-only. Evidence: N0
        stage in the spreadsheet, a separate node structure existing, or the mask
        being one piece rather than several.
  - [ ] Fill the contour and fix the axis order (rt-utils gives rows × cols × slices,
        SimpleITK wants slices × rows × cols).
  - [ ] Save `masks\<id>.nii.gz` with the CT's spacing, origin and direction.
  - [ ] Write a review table: patient, structure chosen, why, number of pieces, size.
  - [ ] Flag empty masks. MIRP fails on them (backlog D4).
- [ ] Look at 3–5 masks over their CTs in a viewer (e.g. 3D Slicer) to confirm
      they line up.

### B. Cohort check
- [ ] Count patients with all three: label + CT + mask.
- [ ] Count deaths in that cohort. With ~60 patients or fewer, a 15% test split is
      ~9 patients, so check it will contain at least a couple of deaths.

### C. Run the existing pipeline on HNSCC
- [ ] `python extract_features_mirp.py` → confirm it still produces **177** features.
      If the count changes, update `Model_GNN(input_dim=...)` in `trainer.py` **and**
      `late_fusion_GNB.py`.
- [ ] `python build_real_csvs.py labels_all.csv` → note the printed split sizes.
- [ ] Delete `dataset\` so the loader rebuilds its cache from the new CSVs.
- [ ] Smoke test first: run the pipeline on a handful of patients before all of them.
- [ ] `python trainer.py gnn`, `python trainer.py cnn`, `python late_fusion_GNB.py`.
      The PC is CPU-only, so the CNN keeps the small (16,32,32) crop for now.

### D. Docs
- [ ] Update README / PROGRESS_SUMMARY: the main dataset is now head-and-neck
      (HNSCC), not lung (backlog H3).
- [ ] Extend the HNSCC section of CHANGELOG as the adapters get finished.

---

## Things to remember

- **Pass real paths, not `.lnk` shortcuts.** Use the shortcut's Target, or
  right-click a file → "Copy as path".
- **Activate the venv** before running anything, on either machine.
- **Per-patient CSVs** (`inventory_details.csv`, `labels_detail.csv`,
  `ct_conversion_report.csv`) contain real IDs. They stay on the PC, and
  `.gitignore` already keeps CSVs out of git.
- **Outcome columns never go in as model inputs**: date of death, cause of death,
  recurrence, follow-up. Feeding them in would let the model cheat.
