"""
convert_ct_hnscc.py - HNSCC DICOM CT -> ct/<Patient_ID>.nii.gz  (CT only; masks come later)

For each patient folder:
  1. find the study whose folder name contains "RT Simulation" (PET studies ignored)
  2. classify its series by the Modality header
  3. pick the CT series:
       - only one CT            -> use it
       - several CTs            -> use the one the RTSTRUCT says it was drawn on
       - several, no reference  -> use the largest, and flag it
  4. read it with SimpleITK and save ct/<Patient_ID>.nii.gz

Run on the Windows PC from the LAMMF folder (REAL path, not the .lnk shortcut):
    python convert_ct_hnscc.py "C:\\...\\patients" --limit 3     # test on 3 first
    python convert_ct_hnscc.py "C:\\...\\patients"               # then everyone

Already-converted patients are skipped, so it's safe to re-run. Use --overwrite
to redo them. Writes ct_conversion_report.csv (stays on this PC).
"""

import os
import sys
import argparse

import pandas as pd
import pydicom
import SimpleITK as sitk


def read_header(fp):
    try:
        return pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    except Exception:
        return None


def find_rt_study(patient_dir):
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


def scan_study(study_dir):
    """Return CT series [(dir, uid, n_files)] and the set of CT UIDs referenced by RTSTRUCTs."""
    cts, refs = [], set()
    for dirpath, _, files in os.walk(study_dir):
        dcms = sorted(f for f in files if f.lower().endswith('.dcm'))
        if not dcms:
            continue
        hdr = read_header(os.path.join(dirpath, dcms[0]))
        mod = str(getattr(hdr, 'Modality', '')) if hdr is not None else ''
        if mod == 'CT':
            cts.append((dirpath, str(getattr(hdr, 'SeriesInstanceUID', '')), len(dcms)))
        elif mod == 'RTSTRUCT':
            for fr in getattr(hdr, 'ReferencedFrameOfReferenceSequence', []):
                for st in getattr(fr, 'RTReferencedStudySequence', []):
                    for se in getattr(st, 'RTReferencedSeriesSequence', []):
                        refs.add(str(se.SeriesInstanceUID))
    return cts, refs


def pick_ct(cts, refs):
    if len(cts) == 1:
        return cts[0], 'only CT'
    referenced = [c for c in cts if c[1] in refs]
    if len(referenced) == 1:
        return referenced[0], 'referenced by RTSTRUCT'
    return max(cts, key=lambda c: c[2]), 'LARGEST (no unique RTSTRUCT reference) - check'


def read_ct(series_dir, uid):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir, uid)   # sorted by slice position
    if not files:
        raise RuntimeError('GDCM found no files for this series UID')
    reader.SetFileNames(files)
    return reader.Execute(), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='REAL path to the folder of patient folders')
    ap.add_argument('--out', default='ct')
    ap.add_argument('--limit', type=int, default=0, help='only the first N patients (testing)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    if a.root.lower().endswith('.lnk'):
        sys.exit("That's a .lnk shortcut. Use the shortcut's Target path instead.")
    if not os.path.isdir(a.root):
        sys.exit(f"Not a folder: {a.root}")
    os.makedirs(a.out, exist_ok=True)

    patients = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    if a.limit:
        patients = patients[:a.limit]

    report = []
    for i, pid in enumerate(patients, 1):
        out_path = os.path.join(a.out, f'{pid}.nii.gz')
        row = {'Patient_ID': pid, 'status': '', 'ct_choice': '', 'slices': '',
               'size_xyz': '', 'spacing_xyz_mm': ''}
        try:
            if os.path.exists(out_path) and not a.overwrite:
                row['status'] = 'already done'
            else:
                studies = find_rt_study(os.path.join(a.root, pid))
                if not studies:
                    row['status'] = 'skipped: no RT Simulation study'
                else:
                    cts, refs = scan_study(studies[0])
                    if not cts:
                        row['status'] = 'skipped: no CT series'
                    else:
                        (sdir, uid, _), why = pick_ct(cts, refs)
                        img, n = read_ct(sdir, uid)
                        sitk.WriteImage(img, out_path)
                        row.update(status='converted', ct_choice=why, slices=n,
                                   size_xyz=str(img.GetSize()),
                                   spacing_xyz_mm=str(tuple(round(s, 2) for s in img.GetSpacing())))
        except Exception as e:
            row['status'] = f'ERROR: {type(e).__name__}: {e}'
        report.append(row)
        print(f"[{i}/{len(patients)}] {row['status']}"
              + (f"  ({row['ct_choice']}, {row['slices']} slices)" if row['ct_choice'] else ''))

    rep = pd.DataFrame(report)
    rep.to_csv('ct_conversion_report.csv', index=False)
    print("\nsummary:")
    print(rep['status'].str.replace(r':.*', '', regex=True).value_counts().to_string())
    flagged = rep['ct_choice'].str.startswith('LARGEST').sum()
    if flagged:
        print(f"  {flagged} patient(s) had several CTs and no clear RTSTRUCT reference - see report")
    print(f"\nCT volumes in: {a.out}/   details: ct_conversion_report.csv (keep on this PC)")


if __name__ == '__main__':
    main()
