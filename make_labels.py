"""
Turn the NSCLC-Radiogenomics clinical CSV into a binary survival label file
(Patient_ID, label) at a chosen time cutoff.

label = 1  -> survived past the cutoff
label = 0  -> died within the cutoff

Usage:
    python make_labels.py NSCLCR01Radiogenomic_DATA_LABELS_...csv --cutoff_days 730
    python make_labels.py <clinical.csv> --cutoff_days 1825 --out labels_5yr.csv

LABELING RULE (documented so you can defend / change it with your advisor):
  - Dead patients: label = 1 if Time to Death (days) > cutoff, else 0.  (clean)
  - Alive patients: labeled 1 (assumed to have survived the window).
      *** CAVEAT: this is the naive choice. An "alive" patient whose follow-up
      is SHORTER than the cutoff is actually "censored" (we don't truly know if
      they'd survive the full window). Proper survival analysis handles this;
      this script does not, because the CSV has no clean follow-up-in-days field
      for alive patients. Decide with your advisor whether to (a) accept this,
      or (b) compute follow-up from the calendar dates and drop censored cases.
  - Patients with no usable status/time are dropped.
"""

import argparse
import numpy as np
import pandas as pd


def build_labels(clinical_csv, cutoff_days, out_csv):
    df = pd.read_csv(clinical_csv, encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]        # some headers have stray spaces

    status = df['Survival Status'].astype(str).str.strip().str.lower()
    ttd = pd.to_numeric(df['Time to Death (days)'], errors='coerce')  # 'N/A' -> NaN

    rows = []
    dead_before = dead_after = alive = dropped = 0
    for pid, st, t in zip(df['Case ID'], status, ttd):
        if st == 'dead':
            if pd.isna(t):
                dropped += 1; continue
            if t > cutoff_days:
                rows.append((pid, 1)); dead_after += 1
            else:
                rows.append((pid, 0)); dead_before += 1
        elif st == 'alive':
            rows.append((pid, 1)); alive += 1            # naive: assume survived (see caveat)
        else:
            dropped += 1

    out = pd.DataFrame(rows, columns=['Patient_ID', 'label'])
    out.to_csv(out_csv, index=False)

    yrs = cutoff_days / 365.0
    print(f"cutoff = {cutoff_days} days (~{yrs:.1f} yr)")
    print(f"  label 1 (survived past cutoff): {(out['label'] == 1).sum()}")
    print(f"  label 0 (died within cutoff):   {(out['label'] == 0).sum()}")
    print(f"  dropped (no usable data):       {dropped}")
    print(f"  [dead-before={dead_before}, dead-after={dead_after}, alive(assumed 1)={alive}]")
    print(f"  wrote {len(out)} labels -> {out_csv}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('clinical_csv')
    ap.add_argument('--cutoff_days', type=int, default=730)   # 730 = 2yr, 1825 = 5yr
    ap.add_argument('--out', default='labels_all.csv')
    a = ap.parse_args()
    build_labels(a.clinical_csv, a.cutoff_days, a.out)
