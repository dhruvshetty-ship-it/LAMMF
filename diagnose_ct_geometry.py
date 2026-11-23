"""
diagnose_ct_geometry.py - why does convert_ct_hnscc.py say "geometry not trusted"?

Takes the first few patients whose status was ERROR in ct_conversion_report.csv
and prints ONLY geometry numbers (orientation, slice positions, spacing, and
what SimpleITK builds from them). No patient IDs, names or dates: patients are
shown as A, B, C. Safe to paste.

Run on the Windows PC, from the LAMMF folder, AFTER convert_ct_hnscc.py finished:
    python diagnose_ct_geometry.py "V:\\QIXR\\TCIA_H&N\\HNSCC"
"""

import os
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk

from convert_ct_hnscc import read_header, find_rt_study, pick_ct, scan_study, series_files, sort_by_position


def fmt(v):
    return '(' + ', '.join(f'{x:8.3f}' for x in v) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    rep = pd.read_csv('ct_conversion_report.csv', dtype=str).fillna('')
    bad = rep[rep['status'].str.startswith('ERROR')]
    print(f"ERROR patients in report: {len(bad)}")
    print("error messages (counts):")
    print(bad['status'].str.replace(r'\d+\.\d+ mm', 'X mm', regex=True).value_counts().to_string())

    for label, pid in zip('ABCDEFGH', bad['Patient_ID'].head(a.n)):
        print(f"\n================ patient {label}")
        studies = find_rt_study(os.path.join(a.root, pid))
        cts, refs = scan_study(studies[0])
        (sdir, uid, _), why = pick_ct(cts, refs)
        gdcm_files = series_files(sdir, uid)
        files, pos = sort_by_position(gdcm_files)
        hdrs = [read_header(f) for f in files]
        iops = np.array([np.array(h.ImageOrientationPatient, float) for h in hdrs])
        ipps = np.array([np.array(h.ImagePositionPatient, float) for h in hdrs])
        row, col = iops[0][:3], iops[0][3:]
        normal = np.cross(row, col)
        print(f"slices: {len(files)}   CT choice: {why}")
        print(f"orientation (row dir, col dir): {fmt(row)} {fmt(col)}")
        print(f"orientation identical on all slices: {np.allclose(iops, iops[0], atol=1e-4)}"
              f"   (max difference {np.abs(iops - iops[0]).max():.5f})")
        print(f"slice normal = row x col: {fmt(normal)}")
        print("first 3 slice positions (sorted):", *[fmt(p) for p in ipps[:3]])
        print("last 3 slice positions (sorted): ", *[fmt(p) for p in ipps[-3:]])
        step = ipps[-1] - ipps[0]
        print(f"first->last: {fmt(step)}   length {np.linalg.norm(step):.2f} mm")
        print(f"  along normal {step @ normal:.2f} mm | sideways along row {step @ row:.2f} mm, "
              f"along col {step @ col:.2f} mm   (sideways != 0 means tilted / drifting slices)")
        gaps = np.diff(pos)
        print(f"gaps along normal: min {gaps.min():.3f} / median {np.median(gaps):.3f} / max {gaps.max():.3f} mm")
        gp = np.array([np.array(read_header(f).ImagePositionPatient, float) @ normal for f in gdcm_files])
        print(f"SimpleITK's own file order runs {'UP (ascending)' if gp[-1] > gp[0] else 'DOWN (descending)'} along the normal")

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        img = reader.Execute()
        d = np.array(img.GetDirection()).reshape(3, 3)
        print(f"SimpleITK built: size {img.GetSize()}, spacing {fmt(img.GetSpacing())}")
        print(f"  origin {fmt(img.GetOrigin())}")
        print(f"  axis directions: x {fmt(d[:, 0])}  y {fmt(d[:, 1])}  z {fmt(d[:, 2])}")
        for i in (0, 1, len(files) - 1):
            got = np.array(img.TransformIndexToPhysicalPoint((0, 0, i)))
            print(f"  slice {i:3d}: SimpleITK puts it at {fmt(got)}, file says {fmt(ipps[i])}, "
                  f"off by {np.linalg.norm(got - ipps[i]):.2f} mm")


if __name__ == '__main__':
    main()
