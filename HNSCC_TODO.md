# HNSCC dataset — to-do log

*Started Oct 9, 2026; last updated Nov 13, 2026 (end of session). Data lives only on the advisor's
Windows PC at `V:\QIXR\TCIA_H&N\HNSCC`. Code goes Mac
(`~/Python/ai_lab/LAMMF_IN_PROGRESS`) → GitHub → Windows.*

## Where things stand

- **Inventory done** (`inventory_data.py`): 65 patient folders, not the expected
  ~130. 61 have an RT Simulation study, 60 have an RTSTRUCT, and every RTSTRUCT
  references a CT in the same study. 2 patients have two CT series. The clinical
  sheet (`Table 1_update`) covers 215 patients, and all 65 imaged patients match
  it by `TCIA code`.
- **CT conversion done** (`convert_ct_hnscc.py`): 61 CTs, 4 skipped (PET only).
  3 had **uneven slice spacing**; all 3 were survivors, so they're dropped into
  `ct_uneven/`. Re-run done on Windows with the new script, so `ct/` should hold
  **58 usable CTs** (confirm: `(Get-ChildItem ct).Count` → 58,
  `(Get-ChildItem ct_uneven).Count` → 3).
- **Labels done** (`make_labels_hnscc.py`, 5-year cutoff, simplified rule L1):
  **55 labeled: 45 survived / 10 died.** (2-year cutoff had given only 4 deaths.)
- **Masks: next.** GTV rules decided (defaults below), so nothing blocks it.

## Decisions made for the masks (backlog H1)

- **Primary GTV only**, no lymph-node GTVs.
- **Pre- vs post-chemo: use post-chemo.** The CT is the radiation-planning scan,
  taken after induction chemo, so the post-chemo outline matches what's on it.
- **The 8 patients with no GTV: dropped (checked Nov 23).** Their structures are
  only organs, planning helpers and CTV/PTV margin volumes; none outlines the
  tumor itself. Several plans reach 70–72 Gy, so for some the tumor was likely
  still in place and a GTV just wasn't drawn. Possible later rescue: use the
  highest-dose CTV, but it's the tumor plus a margin, so it would be inconsistent
  with the other masks.
- **Plain `gtv` (34 patients)**: treat as primary unless evidence says it includes
  nodes (N stage > 0 with no separate node structure, and a mask in several pieces).
- Advisor: tell them these defaults as a heads-up; don't wait on an answer.

---

## Done so far
- [x] Inventory (65 folders, ~60 usable)
- [x] Labels at 5-year cutoff: 55 labeled, 45 survived / 10 died
- [x] CTs converted: 58 usable in `ct\`, 3 uneven moved to `ct_uneven\`, 4 PET-only skipped
- [x] Uneven scans decided (backlog H4): dropped, all 3 were survivors
- [x] GTV rules decided (see "Decisions made for the masks" above)

## Next up

### 1. Quick checks at the start of next session
- [ ] Windows: confirm `(Get-ChildItem ct).Count` = 58 and `ct_uneven` = 3.
- [x] Windows: looked at the 8 no-GTV patients' `roi_names` → no tumor outline, dropped.
- [ ] Mac (only needed to run code there): the project folder moved to
      `~/Python/ai_lab/LAMMF_IN_PROGRESS`, so rebuild the venv (`conda deactivate`
      first): `rm -rf .venv` → `python -m venv .venv` → `source .venv/bin/activate` →
      `python -m pip install --upgrade pip` → `python -m pip install torch` →
      `python -m pip install -r requirements.txt`

### 2. Message your advisor
- [ ] Is 65 folders the full dataset? The spreadsheet has 215 patients; the other
      ~150 patients' scans would roughly triple the deaths (only 10 now).
- [ ] Heads-up on the mask defaults: primary GTV only, post-chemo GTV, drop the
      8 with no GTV.
- [ ] OK to use the simplified label rule for now (cancer deaths only, other deaths
      removed, 5-year cutoff)? Should "Other Cancer" deaths (2 patients) count?
- [ ] Mention the 3 uneven-slice scans that were dropped.

---

## Before training can run

### A. Masks — the next piece of work
- [x] `rt-utils` installed on Windows (Nov 23). It pulled in OpenCV, which upgraded
      numpy to 2.x; fixed by holding OpenCV below 4.12 with `numpy<2`. (Later
      dropped: the mask script no longer uses rt-utils or OpenCV.)
- [x] `convert_rtstruct_hnscc.py` written and tested on fake data (Nov 23).
- [x] First full run (rt-utils version): 23 converted, 9 skipped, 26 errors.
      23 = "CT slice has no RTSTRUCT image slice" (re-matching rt-utils' slices
      to the CT failed), 3 = rt-utils crash on outlines without slice links.
- [x] Rewritten to fill contours directly on the saved CT's grid (pixel counts if
      its center is inside the outline). No rt-utils/OpenCV. Fake tests exact.
      Added `--retry_errors`.
- [ ] **Re-run everyone with `--overwrite`** so all masks come from the same method,
      then check the summary and the flagged patients in `masks_report.csv`.
- [ ] Checklist the script covers (kept for reference): If not, use `platipy` or
      fill the contours ourselves with `scikit-image`. Add the winner to `requirements.txt`.
- [ ] Write `convert_rtstruct_hnscc.py`:
  - [ ] Only for patients with a CT in `ct\` (the 58); build each mask on that CT's grid.
  - [ ] Choose the primary GTV per patient by name. Clear primaries: `gtv primary`,
        `gtv bot`, `gtv tonsil`. Exclude node structures and `ptv70_nogtv`.
        Where pre- and post-chemo GTVs both exist, take post-chemo.
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
