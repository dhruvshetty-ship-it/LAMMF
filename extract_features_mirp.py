"""
Extract IBSI-standard radiomics features with MIRP -- a VALIDATED, actively
maintained, pip-installable radiomics package (the practical alternative to
PyRadiomics, which won't install on modern Python / Apple Silicon).

For each patient: ct/<id>.nii.gz + masks/<id>.nii.gz -> one feature row.
Writes radiomics_features.csv (Patient_ID + ~177 IBSI feature columns), which
build_real_csvs.py then loads into normalized_data_*.csv for the GNN.

Install (works on your Mac / Python 3.13):
    python -m pip install mirp

Run from your LAMMF folder (where ct/ and masks/ live):
    python extract_features_mirp.py
"""

import os
import glob
import pandas as pd
from mirp import extract_features

# columns MIRP adds that are metadata, not features
META_PREFIXES = ('sample_', 'image_', 'mask_', 'settings', 'dir_')


def main():
    rows = []
    for ct_path in sorted(glob.glob('ct/*.nii.gz')):
        pid = os.path.basename(ct_path).replace('.nii.gz', '')
        mask_path = os.path.join('masks', f'{pid}.nii.gz')
        if not os.path.exists(mask_path):
            print(f"  {pid}: no mask, skip")
            continue
        try:
            result = extract_features(
                image=ct_path,
                mask=mask_path,
                new_spacing=2.0,   # resample to 2mm isotropic: IBSI-standard AND much faster
                base_discretisation_method="fixed_bin_number",
                base_discretisation_n_bins=32,
            )
        except Exception as e:
            print(f"  {pid}: extraction failed ({e}) -- skipped")
            continue
        # empty/invalid mask -> MIRP returns None; skip that patient, don't crash
        df = result[0] if result else None
        if df is None or getattr(df, 'empty', True):
            print(f"  {pid}: empty or invalid mask -- skipped")
            continue
        feat_cols = [c for c in df.columns if not c.startswith(META_PREFIXES)]
        row = {'Patient_ID': pid}
        for c in feat_cols:
            row[c] = df.iloc[0][c]
        rows.append(row)
        print(f"  {pid}: {len(feat_cols)} features")

    if not rows:
        print("no features extracted -- are ct/ and masks/ populated?")
        return

    out = pd.DataFrame(rows)
    # keep Patient_ID + numeric feature columns only, fill any gaps with 0
    feat_cols = [c for c in out.columns if c != 'Patient_ID']
    out[feat_cols] = out[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out.to_csv('radiomics_features.csv', index=False)
    print(f"wrote radiomics_features.csv: {len(out)} patients, {len(feat_cols)} features")


if __name__ == '__main__':
    main()
