"""
make_labels_hnscc.py - HNSCC clinical spreadsheet -> labels_all.csv (Patient_ID,label)

SIMPLIFIED LABEL RULE (temporary - see ISSUES_BACKLOG.md, item L1):
  - Alive patients are kept.
  - Patients who died of a CANCER-RELATED cause are kept, as deaths.
  - Patients who died of anything else (non-cancer, unknown, other) are REMOVED.
  Then, at the cutoff (default 2 years = 730 days):
    0 = died (cancer-related) on or before the cutoff
    1 = followed past the cutoff
    removed = alive but followed for less than the cutoff (outcome unknown)

"Cancer-related" = the Cause of Death text mentions "cancer" and isn't "non-cancer".
The script prints how every cause category was classified - check it.

Run on the Windows PC from the LAMMF folder (REAL paths, not .lnk shortcuts):
    python make_labels_hnscc.py "C:\\...\\clinical.xlsx" --patients_root "C:\\...\\patients"

Writes:
  labels_all.csv     Patient_ID,label  (what the rest of the pipeline reads)
  labels_detail.csv  per-patient status / days / cause / reason (keep on this PC)
Printed output has counts and category names only - no patient IDs or dates.
"""

import os
import re
import sys
import argparse

import pandas as pd


def norm(s):
    """Lowercase + collapse whitespace, so 'Survival  (months)' matches 'survival (months)'."""
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def find_col(df, wanted):
    for c in df.columns:
        if norm(c) == norm(wanted):
            return c
    sys.exit(f"Column '{wanted}' not found. Columns are:\n  " + "\n  ".join(map(str, df.columns)))


def is_cancer_cause(cause):
    c = norm(cause)
    return 'cancer' in c and 'non' not in c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clinical', help='clinical Excel (.xlsx) or CSV')
    ap.add_argument('--patients_root', help='REAL path to patient folders; keeps only patients with images')
    ap.add_argument('--cutoff_days', type=int, default=730)        # 730 = 2 yr, 1825 = 5 yr
    ap.add_argument('--sheet', default=0, help='sheet name or index (default: first sheet)')
    ap.add_argument('--out', default='labels_all.csv')
    a = ap.parse_args()

    if a.clinical.lower().endswith(('.xlsx', '.xls')):
        sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
        df = pd.read_excel(a.clinical, sheet_name=sheet)
    else:
        df = pd.read_csv(a.clinical, encoding='latin-1')

    c_id = find_col(df, 'TCIA code')
    c_status = find_col(df, 'Alive or Dead')
    c_days = find_col(df, 'Follow up duration (day)')
    c_cause = find_col(df, 'Cause of Death')

    # drop blank / footer rows (no patient ID)
    df = df[df[c_id].notna()].copy()
    df[c_id] = df[c_id].astype(str).str.strip()
    df = df[df[c_id].str.upper().str.startswith('HNSCC')]
    print(f"patients in spreadsheet: {len(df)}")

    if a.patients_root:
        if a.patients_root.lower().endswith('.lnk'):
            sys.exit("That's a .lnk shortcut. Use the shortcut's Target path instead.")
        imaged = {d for d in os.listdir(a.patients_root)
                  if os.path.isdir(os.path.join(a.patients_root, d))}
        df = df[df[c_id].isin(imaged)].copy()
        print(f"  ...that also have an image folder: {len(df)} (of {len(imaged)} folders)")

    status = df[c_status].astype(str).str.strip().str.lower()
    is_dead = status.str.startswith('dead') | status.str.startswith('deceased')
    print(f"'{c_status}' values found: {sorted(status.unique())}")

    cause = df[c_cause].fillna('(blank)').astype(str).str.strip()
    days = pd.to_numeric(df[c_days], errors='coerce')
    cut = a.cutoff_days

    labels, reasons = [], []
    for dead, cz, d in zip(is_dead, cause, days):
        if dead and not is_cancer_cause(cz):
            labels.append(None); reasons.append('removed: died, cause not cancer-related')
        elif pd.isna(d):
            labels.append(None); reasons.append('removed: missing follow-up time')
        elif dead and d <= cut:
            labels.append(0); reasons.append('label 0: cancer death before cutoff')
        elif d >= cut:
            labels.append(1); reasons.append('label 1: followed past cutoff')
        else:
            labels.append(None); reasons.append('removed: alive, followed < cutoff')
    df['label'] = labels
    df['reason'] = reasons

    # ---------------- summary (counts only)
    print("\nhow each cause-of-death category was treated (dead patients only):")
    for cz in sorted(cause[is_dead].unique()):
        n = int(((cause == cz) & is_dead).sum())
        kind = 'CANCER-RELATED -> kept' if is_cancer_cause(cz) else 'other -> removed'
        print(f"  {n:4d}  {cz:40s} {kind}")

    print(f"\ncutoff: {cut} days")
    print(df['reason'].value_counts().to_string())
    kept = df[df['label'].notna()]
    n1 = int((kept['label'] == 1).sum())
    n0 = int((kept['label'] == 0).sum())
    print(f"\nLABELED: {len(kept)} patients  ->  {n1} survived (1) / {n0} died (0)  "
          f"[{n0 / max(len(kept), 1):.0%} died]")

    # ---------------- write
    out = kept[[c_id, 'label']].rename(columns={c_id: 'Patient_ID'})
    out['label'] = out['label'].astype(int)
    out.to_csv(a.out, index=False)
    df[[c_id, c_status, c_days, c_cause, 'label', 'reason']].rename(
        columns={c_id: 'Patient_ID'}).to_csv('labels_detail.csv', index=False)
    print(f"\nwrote {a.out} ({len(out)} rows) and labels_detail.csv (keep on this PC)")


if __name__ == '__main__':
    main()
