# HNSCC dataset — to-do log

*Started Oct 9, 2026; last updated Nov 23, 2026 (end of session). Data lives only
on the advisor's Windows PC at `V:\QIXR\TCIA_H&N\HNSCC`. Code goes Mac
(`~/Python/ai_lab/LAMMF_IN_PROGRESS`) → GitHub → Windows (`V:\Dhruv_Shetty\LAMMF`).*

## Where things stand (Nov 23) — DATA SETUP IS DONE

All three inputs the pipeline needs exist on the Windows PC:

| Input | Script | Result |
|---|---|---|
| Labels `labels_all.csv` | `make_labels_hnscc.py` | 55 labeled at a 5-year cutoff: 45 survived / 10 died |
| CTs `ct\` | `convert_ct_hnscc.py` | 58 usable; 2 uneven → `ct_uneven\`; 1 error (formerly uneven, dropped); 4 PET-only skipped |
| Masks `masks\` | `convert_rtstruct_hnscc.py` | 48 converted, 10 skipped (no tumor outline), 0 errors |

All masks average soft-tissue CT values (−25 to 93 HU), so no sign of misalignment.
The mask run is also an independent check that the CTs are now correct.

**Today's checkpoint was "run training without errors." Not reached yet:** the
full-pipeline run below was ready but not started (out of time).

---

## NEXT SESSION: run the full pipeline

On Windows, in `V:\Dhruv_Shetty\LAMMF`, venv active (`.\venv\Scripts\Activate.ps1`),
after `git pull`. Paste this whole block. It stops at the first failing step and
saves each step's output in `logs\`.

```
& {
    New-Item -ItemType Directory -Force logs | Out-Null

    Write-Host "`n===== 1/5 radiomics features (slow) =====" -ForegroundColor Cyan
    python extract_features_mirp.py | Tee-Object logs\run_1_features.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at step 1" -ForegroundColor Red; return }

    Write-Host "`n===== 2/5 build CSVs + split =====" -ForegroundColor Cyan
    python build_real_csvs.py labels_all.csv | Tee-Object logs\run_2_build.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at step 2" -ForegroundColor Red; return }

    Remove-Item -Recurse -Force dataset -ErrorAction SilentlyContinue

    Write-Host "`n===== 3/5 train GNN =====" -ForegroundColor Cyan
    python trainer.py gnn | Tee-Object logs\run_3_gnn.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at step 3" -ForegroundColor Red; return }

    Write-Host "`n===== 4/5 train CNN =====" -ForegroundColor Cyan
    python trainer.py cnn | Tee-Object logs\run_4_cnn.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at step 4" -ForegroundColor Red; return }

    Write-Host "`n===== 5/5 fusion =====" -ForegroundColor Cyan
    python late_fusion_GNB.py | Tee-Object logs\run_5_fusion.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at step 5" -ForegroundColor Red; return }

    Write-Host "`nALL 5 STEPS FINISHED WITHOUT ERRORS" -ForegroundColor Green
}
```

Then show the results (safe to paste):
```
Get-Content logs\run_1_features.txt -Tail 2; Get-Content logs\run_2_build.txt; Get-Content logs\run_5_fusion.txt
```

What to check:
- [ ] Step 1 says **177 features** (if not: update `Model_GNN(input_dim=...)` in
      `trainer.py` AND `late_fusion_GNB.py`). Slow: tens of minutes to ~1 hour.
- [ ] Step 2: **every split (train / validation / test) has at least 1 died.**
      If it errors, it's likely too few deaths for the stratified split.
- [ ] Steps 3–5 finish → **checkpoint met.** Don't judge the AUC numbers: with
      ~1–2 deaths in the test set they're noisy. This run proves the pipeline works.

---

## Open questions / advisor

- [ ] Is 65 folders the full dataset? The spreadsheet has 215 patients; the other
      ~150 patients' scans would roughly triple the deaths (only 10 now).
- [ ] **Masks: include all visible cancer (primary + lymph nodes)?** (backlog H5,
      leaning yes). Explain the current inconsistency below.
- [ ] OK to use the simplified label rule for now (cancer deaths only, other deaths
      removed, 5-year cutoff)? Should "Other Cancer" deaths (2 patients) count?
- [ ] Mention the 3 uneven-slice scans dropped, and the 8 patients with no GTV.

## Future changes (logged, not done)

- [ ] **H5 — masks = all visible cancer.** Today's masks are primary-GTV only, which
      is inconsistent: patients with separate `gtv primary` + `gtv nodes` get the
      primary only, but a plain `gtv` (29 of 48) often already includes nodes (16
      masks come out in several similar-sized pieces). Node spread is strongly
      prognostic. Plan: `--include_nodes` option (default on) that merges all GTV
      structures; keep primary-only as an option and compare. Longer term: primary
      and each node as separate regions → separate GNN graph nodes (D5/M4).
- [ ] Patient flagged "ONLY PRE-CHEMO" with 5 tiny pieces (1.6 cc, −15 HU) looks
      wrong; consider dropping it.
- [ ] Stop flagging plain-`gtv` multi-piece masks when the patient also has a
      separate node structure (then the pieces aren't nodes).
- [ ] Visual check: a script that saves a picture of each CT with its mask outline,
      to confirm masks sit on the tumor and to sort multi-piece masks.
- [ ] Mac venv: the project folder moved, so rebuild `.venv` before running code on
      the Mac (`conda deactivate` → `rm -rf .venv` → `python -m venv .venv` →
      `source .venv/bin/activate` → `python -m pip install --upgrade pip` →
      `python -m pip install torch` → `python -m pip install -r requirements.txt`).
- [ ] Update README / PROGRESS_SUMMARY wording: main dataset is now HNSCC (H3).

---

## How we got here (decisions and bugs, for the record)

**Data**
- Inventory: 65 patient folders (~60 usable); clinical sheet `Table 1_update`
  (215 patients) matches all 65 by `TCIA code`.
- Labels: 2-year cutoff gave only 4 deaths → switched to **5 years** (10 deaths).
  Simplified rule (L1): alive + cancer-related deaths kept, other deaths removed,
  alive-but-followed-<5-years removed.
- Masks: **primary GTV only** for now; post-chemo over pre-chemo; nodes and
  CTV/PTV excluded; empty placeholder structures ignored. 8 patients have no tumor
  outline at all (only organs and CTV/PTV margins) → dropped.

**Bugs found and fixed (Nov 23)**
1. *Uneven slice spacing* (3 scans; one had a 1.1 m gap = two scans in one series):
   SimpleITK averages the gaps, distorting the volume → saved to `ct_uneven\`,
   excluded.
2. *rt-utils masks* (first mask version): 23 failed re-matching its slices to the
   CT, 3 crashed on outlines without slice links → rewrote the mask script to fill
   contours directly on the CT grid (a pixel counts if its center is inside the
   outline). No rt-utils/OpenCV needed.
3. *Flipped CTs*: "contour lands at slice position −N". Root cause found with
   `diagnose_ct_geometry.py`: the scans are perfectly regular, but **SimpleITK
   labels the slice direction as pointing down** on many HNSCC scans → the CT
   script now sorts slices by position, sets the geometry itself from the DICOM
   headers, and checks every slice is within 1 mm of its file. Tested with a
   simulated mislabel that reproduces the PC's numbers exactly.
4. *Stale files*: errored patients used to keep an old output file → both scripts
   now delete it.
5. *numpy 2 via OpenCV* (when rt-utils was installed): fixed with `numpy<2` and
   OpenCV < 4.12; now moot since nothing uses them.

## Things to remember

- **Pass real paths, not `.lnk` shortcuts.** Use the shortcut's Target, or
  right-click a file → "Copy as path". Clinical file:
  `V:\QIXR\TCIA_H&N\HNSCC\HNSCC-ClinData_date_offset.xlsx`.
- **Activate the venv** before running anything, on either machine.
- **If `git pull` refuses** because of local changes: `git stash`, then `git pull`.
- **Per-patient CSVs** (`inventory_details.csv`, `labels_detail.csv`,
  `ct_conversion_report.csv`, `masks_report.csv`) contain real IDs. They stay on
  the PC; `.gitignore` keeps CSVs out of git. The printed summaries are ID-free.
- **Outcome columns never go in as model inputs**: date of death, cause of death,
  recurrence, follow-up. Feeding them in would let the model cheat.
- Useful flags: `--limit N` (test on a few), `--overwrite` (redo), and for masks
  `--retry_errors` (redo only failures), `--clinical` (adds N stage to the report).
