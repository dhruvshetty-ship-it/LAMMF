"""
convert_ct_hnscc.py - HNSCC DICOM CT -> ct/<Patient_ID>.nii.gz  (CT only; masks come later)

For each patient folder:
  1. find the study whose folder name contains "RT Simulation" (PET studies ignored)
  2. classify its series by the Modality header
  3. pick the CT series:
       - only one CT            -> use it
       - several CTs            -> use the one the RTSTRUCT says it was drawn on
       - several, no reference  -> use the largest, and flag it
  4. sort the slices by their physical position (so the saved volume can't be
     flipped top-to-bottom) and check slice spacing (flags missing slices /
     mixed slice thickness)
  5. read the pixels with SimpleITK, set the geometry (origin, orientation,
     spacing) ourselves from the DICOM headers - SimpleITK's own slice direction
     is wrong on many HNSCC scans - check every slice lands where its file says,
     and save ct/<Patient_ID>.nii.gz
     - UNEVEN scans are saved to ct_uneven/ instead, so the rest of the pipeline
       skips them (an uneven scan left in ct/ by an older run gets moved there)

Run on the Windows PC from the LAMMF folder (REAL path, not the .lnk shortcut):
    python convert_ct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --limit 3     # test on 3 first
    python convert_ct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC"               # then everyone

Already-converted patients are skipped (but still spacing-checked), so it's safe
to re-run. Use --overwrite to redo them. Writes ct_conversion_report.csv (stays
on this PC).
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

# ITK prints its own "non uniform sampling" warning; our spacing check below
# reports the same thing per patient, with numbers, so silence the duplicate.
sitk.ProcessObject.SetGlobalWarningDisplay(False)


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


def series_files(series_dir, uid):
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, uid)  # sorted by position
    if not files:
        raise RuntimeError('GDCM found no files for this series UID')
    return files


def sort_by_position(files):
    """
    Sort slices by their physical position along the slice-stacking direction
    (the cross product of the image row and column directions).
    SimpleITK labels the volume as running along that direction, so the files
    must be in that order too. Its own file sorting can fall back to e.g. the
    instance number, which may run the other way and silently flip the scan.
    Returns the sorted files and their sorted positions (mm).
    """
    pos = []
    for f in files:
        h = read_header(f)
        ipp = np.array(h.ImagePositionPatient, dtype=float)
        iop = np.array(h.ImageOrientationPatient, dtype=float)
        pos.append(float(ipp @ np.cross(iop[:3], iop[3:])))
    order = np.argsort(pos)
    return [files[i] for i in order], np.array(pos)[order]


def slice_gaps(pos_sorted):
    """
    Distances (mm) between consecutive slices. A normal scan has (nearly)
    identical gaps; a missing slice or mixed slice thickness shows up as one or
    more gaps that differ from the rest.
    """
    gaps = np.diff(pos_sorted)
    med = float(np.median(gaps))
    n_off = int((np.abs(gaps - med) > 0.1).sum())    # gaps more than 0.1 mm from typical
    return float(gaps.min()), float(gaps.max()), med, n_off


def read_ct(files, pos, strict=True):
    """
    Read the pixel data with SimpleITK (slice i = files[i], already sorted by
    position), then set the geometry OURSELVES from the DICOM headers.
    SimpleITK's own z-direction can't be trusted on this data: on many HNSCC
    scans it labels the slice direction as pointing down while the slices go up,
    which flips every position along the body.

    Safety check: every saved slice must sit within 1 mm of where its own DICOM
    file says it is (strict=True), or just the first and last slice (strict=False,
    used for uneven scans, which can't be stored exactly anyway).
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()

    h0 = read_header(files[0])
    iop = np.array(h0.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ps = [float(v) for v in h0.PixelSpacing]            # DICOM: [between rows, between columns]
    nz = img.GetSize()[2]
    dz = (pos[-1] - pos[0]) / (nz - 1) if nz > 1 else float(getattr(h0, 'SliceThickness', 1.0))
    img.SetOrigin(tuple(float(v) for v in h0.ImagePositionPatient))
    img.SetSpacing((ps[1], ps[0], float(dz)))           # x = along a row, y = down a column, z = between slices
    img.SetDirection(tuple(np.column_stack([row_dir, col_dir, normal]).ravel()))

    ipps = [np.array(read_header(f).ImagePositionPatient, dtype=float) for f in files]
    for idx in (range(nz) if strict else (0, nz - 1)):
        got = np.array(img.TransformIndexToPhysicalPoint((0, 0, idx)))
        off = np.linalg.norm(got - ipps[idx])
        if off > 1.0:
            raise RuntimeError(f'saved slice {idx} would be {off:.1f} mm away from its DICOM '
                               f'position - geometry not trusted')
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='REAL path to the folder of patient folders')
    ap.add_argument('--out', default='ct')
    ap.add_argument('--uneven_out', default='ct_uneven',
                    help='uneven-slice scans go here instead, so the pipeline skips them')
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
               'size_xyz': '', 'spacing_xyz_mm': '', 'gap_min_mm': '', 'gap_max_mm': '',
               'gap_typical_mm': '', 'n_uneven_gaps': '', 'spacing_check': ''}
        try:
            studies = find_rt_study(os.path.join(a.root, pid))
            if not studies:
                row['status'] = 'skipped: no RT Simulation study'
            else:
                cts, refs = scan_study(studies[0])
                if not cts:
                    row['status'] = 'skipped: no CT series'
                else:
                    (sdir, uid, _), why = pick_ct(cts, refs)
                    files, pos = sort_by_position(series_files(sdir, uid))
                    gmin, gmax, gmed, n_off = slice_gaps(pos)
                    row.update(ct_choice=why, slices=len(files), gap_min_mm=round(gmin, 2),
                               gap_max_mm=round(gmax, 2), gap_typical_mm=round(gmed, 2),
                               n_uneven_gaps=n_off,
                               spacing_check='UNEVEN' if gmax - gmin > 0.5 else 'ok')
                    uneven = row['spacing_check'] == 'UNEVEN'
                    uneven_path = os.path.join(a.uneven_out, f'{pid}.nii.gz')
                    dest = uneven_path if uneven else out_path
                    if uneven and os.path.exists(out_path):
                        # converted by an older run into ct/ - move it out of the pipeline
                        os.makedirs(a.uneven_out, exist_ok=True)
                        os.replace(out_path, uneven_path)
                        row['status'] = f'moved to {a.uneven_out} (uneven)'
                    elif os.path.exists(dest) and not a.overwrite:
                        row['status'] = 'already done' + (f' ({a.uneven_out})' if uneven else '')
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        img = read_ct(files, pos, strict=not uneven)
                        sitk.WriteImage(img, dest)
                        row.update(status='converted' + (f' to {a.uneven_out} (uneven)' if uneven else ''),
                                   size_xyz=str(img.GetSize()),
                                   spacing_xyz_mm=str(tuple(round(s, 2) for s in img.GetSpacing())))
        except Exception as e:
            row['status'] = f'ERROR: {type(e).__name__}: {e}'
        if row['status'].startswith(('ERROR', 'skipped')):
            for stale in (out_path, os.path.join(a.uneven_out, f'{pid}.nii.gz')):
                if os.path.exists(stale):
                    os.remove(stale)     # don't leave an old (possibly wrong) CT from an earlier run
        report.append(row)
        msg = f"[{i}/{len(patients)}] {row['status']}"
        if row['ct_choice']:
            msg += f"  ({row['ct_choice']}, {row['slices']} slices)"
        if row['spacing_check'] == 'UNEVEN':
            msg += (f"  UNEVEN SLICES: gaps {row['gap_min_mm']}-{row['gap_max_mm']} mm, "
                    f"typical {row['gap_typical_mm']}, {row['n_uneven_gaps']} odd gap(s)")
        print(msg)

    rep = pd.DataFrame(report)
    rep.to_csv('ct_conversion_report.csv', index=False)
    print("\nsummary:")
    print(rep['status'].str.replace(r':.*', '', regex=True).value_counts().to_string())
    flagged = rep['ct_choice'].str.startswith('LARGEST').sum()
    if flagged:
        print(f"  {flagged} patient(s) had several CTs and no clear RTSTRUCT reference - see report")
    uneven = rep[rep['spacing_check'] == 'UNEVEN']
    print(f"\nslice spacing: {int((rep['spacing_check'] == 'ok').sum())} even, {len(uneven)} UNEVEN "
          f"(largest gap minus smallest gap > 0.5 mm)")
    if len(uneven):
        print("  uneven scans (typical gap -> largest gap, number of odd gaps):")
        for _, r in uneven.iterrows():
            print(f"    {r['gap_typical_mm']} -> {r['gap_max_mm']} mm, {r['n_uneven_gaps']} odd gap(s)")
    print(f"\nCT volumes in: {a.out}/   details: ct_conversion_report.csv (keep on this PC)")


if __name__ == '__main__':
    main()
