"""
inventory_data.py - read-only recon of a TCIA-style radiotherapy dataset (e.g. HNSCC).

Expected layout (per patient):
    <root>/HNSCC-01-0068/<...RT Simulation...>/<series folder>/*.dcm
PET studies are ignored; only studies whose folder name contains "RT Simulation"
are inspected.

Run ON THE WINDOWS PC (the data never leaves it). Pass the REAL folder path,
not the .lnk shortcut (right-click shortcut > Properties > copy "Target"):

    python inventory_data.py "D:\\real\\path\\to\\patients" --clinical "D:\\path\\to\\clinical.xlsx"

Outputs:
  - Printed summary: counts, modalities, structure names, spreadsheet column
    names. No patient names, dates or IDs, so it is safe to paste elsewhere.
  - inventory_details.csv: one row per patient WITH real IDs, for your own
    debugging. Stays on this PC (.gitignore already excludes *.csv).

Nothing is modified; the script only reads DICOM headers.
"""

import os
import sys
import argparse
import statistics
from collections import Counter

import pandas as pd
import pydicom


# ---------------------------------------------------------------- Windows helpers
def win_long(path):
    """Windows refuses paths over 260 chars by default; the \\\\?\\ prefix lifts that."""
    path = os.path.abspath(path)
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        path = '\\\\?\\' + path
    return path


def plain(path):
    """Strip the long-path prefix, for measuring/printing."""
    return path[4:] if path.startswith('\\\\?\\') else path


# ---------------------------------------------------------------- DICOM helpers
def read_header(fp):
    """Read a DICOM header without pixel data (fast). None if unreadable."""
    try:
        return pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    except Exception:
        return None


def find_rt_studies(patient_dir):
    """All folders under a patient whose name contains 'RT Simulation'."""
    hits = []
    for dirpath, dirnames, _ in os.walk(patient_dir):
        keep = []
        for d in dirnames:
            if 'rt simulation' in d.lower():
                hits.append(os.path.join(dirpath, d))
            else:
                keep.append(d)
        dirnames[:] = keep            # don't descend into a matched study
    return sorted(hits)


def inspect_study(study_dir):
    """Classify every series in a study by its Modality header."""
    info = {'ct': [], 'rtstruct': [], 'modalities': Counter(), 'max_path': 0}
    for dirpath, _, files in os.walk(study_dir):
        for f in files:
            info['max_path'] = max(info['max_path'], len(plain(os.path.join(dirpath, f))))
        dcms = sorted(f for f in files if f.lower().endswith('.dcm'))
        if not dcms:
            continue
        hdr = read_header(os.path.join(dirpath, dcms[0]))
        mod = str(getattr(hdr, 'Modality', 'UNREADABLE')) if hdr is not None else 'UNREADABLE'
        info['modalities'][mod] += 1

        if mod == 'CT':
            info['ct'].append({'uid': str(getattr(hdr, 'SeriesInstanceUID', '')),
                               'n_slices': len(dcms)})
        elif mod == 'RTSTRUCT':
            for f in dcms:                      # normally exactly one file
                rs = read_header(os.path.join(dirpath, f))
                if rs is None:
                    continue
                names = [str(r.ROIName).strip()
                         for r in getattr(rs, 'StructureSetROISequence', [])]
                refs = set()                    # which CT series the contours were drawn on
                for fr in getattr(rs, 'ReferencedFrameOfReferenceSequence', []):
                    for st in getattr(fr, 'RTReferencedStudySequence', []):
                        for se in getattr(st, 'RTReferencedSeriesSequence', []):
                            refs.add(str(se.SeriesInstanceUID))
                info['rtstruct'].append({'names': names, 'refs': refs})
    return info


# ---------------------------------------------------------------- clinical sheet
def summarize_clinical(path, patient_ids):
    print("\n=== CLINICAL FILE (column names and counts only) ===")
    if path.lower().endswith(('.xlsx', '.xls')):
        sheets = pd.read_excel(path, sheet_name=None)        # needs: pip install openpyxl
    else:
        sheets = {'csv': pd.read_csv(path, encoding='latin-1')}

    for sheet, df in sheets.items():
        print(f"\n--- sheet '{sheet}': {len(df)} rows x {df.shape[1]} columns")
        id_col = None
        for c in df.columns:
            if df[c].astype(str).str.contains('HNSCC-', na=False).mean() > 0.5:
                id_col = c
                break
        for c in df.columns:
            s = df[c]
            line = (f"  {str(c)[:45]:45s} {str(s.dtype):9s} "
                    f"non-null {s.notna().sum():4d}  unique {s.nunique():4d}")
            # show category values only for small, non-date text columns (e.g. Alive/Dead)
            if (s.dtype == object and s.nunique() <= 8 and c != id_col
                    and 'date' not in str(c).lower()):
                line += f"  values: {sorted(map(str, s.dropna().unique()))}"
            print(line)
        if id_col is not None:
            ids = set(df[id_col].astype(str).str.strip())
            found = sum(p in ids for p in patient_ids)
            print(f"  patient ID column appears to be '{id_col}': "
                  f"{found}/{len(patient_ids)} image patients appear in this sheet")
        else:
            print("  (no column of HNSCC-style IDs found in this sheet)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='REAL path to the folder of patient folders (not a .lnk shortcut)')
    ap.add_argument('--clinical', help='clinical Excel/CSV file (optional)')
    ap.add_argument('--out', default='inventory_details.csv')
    a = ap.parse_args()

    if a.root.lower().endswith('.lnk'):
        sys.exit("That's a Windows shortcut (.lnk). Python can't follow those.\n"
                 "Right-click it > Properties > copy the 'Target' path and pass that instead.")
    root = win_long(a.root)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {a.root}")

    patients = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"patient folders found: {len(patients)}  (reading headers, this can take a minute)")

    rows = []
    roi_counts = Counter()           # structure name -> number of patients that have it
    all_modalities = Counter()
    slice_counts = []
    max_path = 0

    for i, pid in enumerate(patients, 1):
        studies = find_rt_studies(os.path.join(root, pid))
        row = {'Patient_ID': pid, 'n_rt_studies': len(studies), 'n_ct_series': 0,
               'ct_slices': 0, 'n_rtstruct': 0, 'n_rtdose': 0, 'n_rtplan': 0,
               'other_modalities': '', 'rtstruct_refs_ct': '', 'gtv_names': '',
               'roi_names': ''}
        if studies:
            info = inspect_study(studies[0])          # >1 RT study is flagged via n_rt_studies
            mods = info['modalities']
            all_modalities.update(mods)
            max_path = max(max_path, info['max_path'])
            row['n_ct_series'] = mods.get('CT', 0)
            row['n_rtstruct'] = len(info['rtstruct'])
            row['n_rtdose'] = mods.get('RTDOSE', 0)
            row['n_rtplan'] = mods.get('RTPLAN', 0)
            row['other_modalities'] = '; '.join(
                f"{m}x{n}" for m, n in mods.items() if m not in ('CT', 'RTSTRUCT', 'RTDOSE', 'RTPLAN'))
            if info['ct']:
                row['ct_slices'] = max(c['n_slices'] for c in info['ct'])
                slice_counts.extend(c['n_slices'] for c in info['ct'])
            if info['rtstruct']:
                names = sorted({n for rs in info['rtstruct'] for n in rs['names']})
                row['roi_names'] = '; '.join(names)
                row['gtv_names'] = '; '.join(n for n in names if 'gtv' in n.lower())
                roi_counts.update({n.lower() for n in names})
                ct_uids = {c['uid'] for c in info['ct']}
                row['rtstruct_refs_ct'] = any(rs['refs'] & ct_uids for rs in info['rtstruct'])
        rows.append(row)
        if i % 10 == 0 or i == len(patients):
            print(f"  ...{i}/{len(patients)}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # ------------------------------------------------ safe-to-share summary
    n = len(df)
    print("\n=== IMAGING SUMMARY (safe to share: no names, dates or IDs) ===")
    print(f"patients: {n}")
    print(f"RT Simulation studies per patient: {dict(Counter(df['n_rt_studies']))}")
    print(f"series by modality (all patients): {dict(all_modalities)}")
    print(f"CT series per patient:  {dict(Counter(df['n_ct_series']))}")
    print(f"RTSTRUCT files per patient: {dict(Counter(df['n_rtstruct']))}")
    if slice_counts:
        print(f"CT slices per series: min {min(slice_counts)} / "
              f"median {statistics.median(slice_counts):.0f} / max {max(slice_counts)}")
    has_rs = df['n_rtstruct'] > 0
    refs = df.loc[has_rs, 'rtstruct_refs_ct']
    print(f"RTSTRUCT points at a CT in the same study: yes {int((refs == True).sum())} / "
          f"no {int((refs == False).sum())}  (of {int(has_rs.sum())} with an RTSTRUCT)")
    print(f"patients with a structure named like 'GTV': {int((df['gtv_names'] != '').sum())}/{n}")
    print(f"longest file path: {max_path} characters (Windows default limit is 260)")

    print("\nproblem counts (details in the CSV):")
    print(f"  no RT Simulation study: {int((df['n_rt_studies'] == 0).sum())}")
    print(f"  more than one RT Simulation study: {int((df['n_rt_studies'] > 1).sum())}")
    print(f"  no CT series: {int(((df['n_rt_studies'] > 0) & (df['n_ct_series'] == 0)).sum())}")
    print(f"  more than one CT series: {int((df['n_ct_series'] > 1).sum())}")
    print(f"  no RTSTRUCT: {int(((df['n_rt_studies'] > 0) & (df['n_rtstruct'] == 0)).sum())}")
    print(f"  RTSTRUCT but no GTV-named structure: {int((has_rs & (df['gtv_names'] == '')).sum())}")

    print("\nstructure names containing 'gtv' (patients with each):")
    for name, cnt in sorted(((k, v) for k, v in roi_counts.items() if 'gtv' in k),
                            key=lambda kv: -kv[1]):
        print(f"  {cnt:4d}  {name}")
    print("\n40 most common structure names overall (patients with each):")
    for name, cnt in roi_counts.most_common(40):
        print(f"  {cnt:4d}  {name}")

    if a.clinical:
        summarize_clinical(a.clinical, patients)

    print(f"\nper-patient details (with real IDs, keep on this PC): {a.out}")


if __name__ == '__main__':
    main()
