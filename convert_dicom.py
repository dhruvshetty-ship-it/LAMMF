"""
Convert a downloaded TCIA patient folder into the model's image inputs:
  DICOM CT series -> ct/<PatientID>.nii.gz
  DICOM-SEG file  -> masks/<PatientID>.nii.gz   (aligned to the CT grid)

Run:
    python convert_dicom.py "/path/to/NSCLC download root" --out .. --require_seg --limit 40

Needs:
    python -m pip install SimpleITK "pydicom==2.4.4" pydicom-seg
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import pydicom


def find_ct_and_seg(patient_dir):
    ct_series = {}
    seg_files = []
    for dirpath, _, files in os.walk(patient_dir):
        dcms = [os.path.join(dirpath, f) for f in files if f.lower().endswith('.dcm')]
        if not dcms:
            continue
        try:
            ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
        except Exception:
            continue
        modality = getattr(ds, 'Modality', '')
        if modality == 'CT':
            ct_series[dirpath] = len(dcms)
        elif modality == 'SEG':
            seg_files.append(dcms[0])
    ct_dir = max(ct_series, key=ct_series.get) if ct_series else None
    seg_file = seg_files[0] if seg_files else None
    return ct_dir, seg_file


def convert_ct(ct_dir, out_path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_dir)
    if series_ids:
        files = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
    else:
        files = reader.GetGDCMSeriesFileNames(ct_dir)
    reader.SetFileNames(files)
    img = reader.Execute()
    sitk.WriteImage(img, out_path)
    return img


def convert_seg(seg_file, ref_img, out_path):
    import pydicom_seg
    ds = pydicom.dcmread(seg_file)
    result = pydicom_seg.SegmentReader().read(ds)
    merged = None
    first_seg_img = None
    for num in result.available_segments:
        seg_img = result.segment_image(num)
        if first_seg_img is None:
            first_seg_img = seg_img
        arr = sitk.GetArrayFromImage(seg_img)
        merged = arr if merged is None else np.maximum(merged, arr)
    mask = sitk.GetImageFromArray((merged > 0).astype(np.uint8))
    mask.CopyInformation(first_seg_img)
    mask = sitk.Resample(mask, ref_img, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(mask, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='root of the NBIA download')
    ap.add_argument('--out', default='.', help='where ct/ and masks/ go')
    ap.add_argument('--limit', type=int, default=0,
                    help='convert at most this many patients (0 = all)')
    ap.add_argument('--require_seg', action='store_true',
                    help='skip patients that have no segmentation')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'ct'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'masks'), exist_ok=True)

    patient_dirs = []
    for dirpath, dirs, _ in os.walk(args.root):
        for d in dirs:
            if d.startswith('R01-') or d.startswith('AMC-'):
                patient_dirs.append(os.path.join(dirpath, d))

    print(f"found {len(patient_dirs)} patient folders")
    ok, skipped = 0, 0
    for pdir in sorted(set(patient_dirs)):
        if args.limit and ok >= args.limit:
            break
        pid = os.path.basename(pdir)
        ct_dir, seg_file = find_ct_and_seg(pdir)
        if ct_dir is None:
            print(f"  {pid}: no CT series found, skipping"); skipped += 1; continue
        if args.require_seg and seg_file is None:
            print(f"  {pid}: no SEG, skipping (--require_seg)"); skipped += 1; continue
        ct_img = convert_ct(ct_dir, os.path.join(args.out, 'ct', f'{pid}.nii.gz'))
        if seg_file is not None:
            try:
                convert_seg(seg_file, ct_img, os.path.join(args.out, 'masks', f'{pid}.nii.gz'))
            except Exception as e:
                print(f"  {pid}: CT ok, SEG failed ({e})")
        else:
            print(f"  {pid}: CT ok, no SEG found")
        ok += 1
    print(f"done: {ok} converted, {skipped} skipped")


if __name__ == '__main__':
    main()