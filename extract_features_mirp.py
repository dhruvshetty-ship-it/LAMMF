"""
Extract IBSI-standard radiomics features with MIRP -- a VALIDATED, actively
maintained, pip-installable radiomics package (the practical alternative to
PyRadiomics, which won't install on modern Python / Apple Silicon).

For each patient: ct/<id>.nii.gz + masks/<id>.nii.gz -> one feature row.
Writes radiomics_features.csv (Patient_ID + ~177 IBSI feature columns), which
build_real_csvs.py then loads into normalized_data_*.csv for the GNN.

SPEED: before MIRP sees a patient, the CT and mask are cropped to a box around
the tumor plus a margin (default 20 mm on every side). The features are only
measured inside the tumor, so this gives the same kind of result while MIRP no
longer resamples the whole 512x512 scan - many times faster on full-size CTs.
Cropping shifts the 2 mm resampling grid slightly, so values can differ a tiny
bit from an uncropped run; every patient is processed the same way, so that's
fine. Use --no_crop to process whole scans (slow).

Run from your LAMMF folder (where ct/ and masks/ live):
    python extract_features_mirp.py
    python extract_features_mirp.py --margin_mm 30      # bigger margin
"""

import os
import glob
import time
import argparse
import tempfile

import numpy as np
import pandas as pd
import SimpleITK as sitk
from mirp import extract_features

# columns MIRP adds that are metadata, not features
META_PREFIXES = ('sample_', 'image_', 'mask_', 'settings', 'dir_')


def crop_to_mask(ct_path, mask_path, margin_mm, out_dir):
    """Crop CT + mask to the mask's bounding box plus a margin. None if the mask is empty."""
    ct = sitk.ReadImage(ct_path)
    mask = sitk.ReadImage(mask_path)
    zz, yy, xx = np.nonzero(sitk.GetArrayFromImage(mask))
    if len(xx) == 0:
        return None
    size = ct.GetSize()                                          # (x, y, z)
    pad = [int(np.ceil(margin_mm / s)) for s in ct.GetSpacing()]
    lo = [max(int(v.min()) - p, 0) for v, p in zip((xx, yy, zz), pad)]
    hi = [min(int(v.max()) + p, n - 1) for v, p, n in zip((xx, yy, zz), pad, size)]
    extent = [h - l + 1 for l, h in zip(lo, hi)]
    ct_c = sitk.RegionOfInterest(ct, extent, lo)                 # keeps the real-world geometry
    mask_c = sitk.RegionOfInterest(mask, extent, lo)
    ct_out, mask_out = os.path.join(out_dir, 'ct.nii.gz'), os.path.join(out_dir, 'mask.nii.gz')
    sitk.WriteImage(ct_c, ct_out)
    sitk.WriteImage(mask_c, mask_out)
    return ct_out, mask_out, tuple(extent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin_mm', type=float, default=20.0)
    ap.add_argument('--no_crop', action='store_true', help='process whole scans (much slower)')
    a = ap.parse_args()

    ct_files = sorted(glob.glob('ct/*.nii.gz'))
    rows = []
    t_all = time.time()
    for i, ct_path in enumerate(ct_files, 1):
        pid = os.path.basename(ct_path).replace('.nii.gz', '')
        mask_path = os.path.join('masks', f'{pid}.nii.gz')
        if not os.path.exists(mask_path):
            print(f"  [{i}/{len(ct_files)}] {pid}: no mask, skip")
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if a.no_crop:
                    img_in, mask_in, box = ct_path, mask_path, 'whole scan'
                else:
                    cropped = crop_to_mask(ct_path, mask_path, a.margin_mm, tmp)
                    if cropped is None:
                        print(f"  [{i}/{len(ct_files)}] {pid}: empty mask -- skipped")
                        continue
                    img_in, mask_in, box = cropped
                result = extract_features(
                    image=img_in,
                    mask=mask_in,
                    new_spacing=2.0,   # resample to 2mm isotropic: IBSI-standard AND much faster
                    base_discretisation_method="fixed_bin_number",
                    base_discretisation_n_bins=32,
                )
            except Exception as e:
                print(f"  [{i}/{len(ct_files)}] {pid}: extraction failed ({e}) -- skipped")
                continue
        # empty/invalid mask -> MIRP returns None; skip that patient, don't crash
        df = result[0] if result else None
        if df is None or getattr(df, 'empty', True):
            print(f"  [{i}/{len(ct_files)}] {pid}: empty or invalid mask -- skipped")
            continue
        feat_cols = [c for c in df.columns if not c.startswith(META_PREFIXES)]
        row = {'Patient_ID': pid}
        for c in feat_cols:
            row[c] = df.iloc[0][c]
        rows.append(row)
        print(f"  [{i}/{len(ct_files)}] {pid}: {len(feat_cols)} features  "
              f"({time.time() - t0:.0f} s, box {box})", flush=True)

    if not rows:
        print("no features extracted -- are ct/ and masks/ populated?")
        return

    out = pd.DataFrame(rows)
    # keep Patient_ID + numeric feature columns only, fill any gaps with 0
    feat_cols = [c for c in out.columns if c != 'Patient_ID']
    out[feat_cols] = out[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out.to_csv('radiomics_features.csv', index=False)
    print(f"total time {(time.time() - t_all) / 60:.1f} min")
    print(f"wrote radiomics_features.csv: {len(out)} patients, {len(feat_cols)} features")


if __name__ == '__main__':
    main()
