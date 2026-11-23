"""
convert_rtstruct_hnscc.py - HNSCC RTSTRUCT primary GTV -> masks/<Patient_ID>.nii.gz

Runs only for patients that already have ct/<Patient_ID>.nii.gz (from
convert_ct_hnscc.py), and builds each mask on exactly that CT's grid.

For each patient:
  1. find the RT Simulation study, the CT series that ct/<id>.nii.gz came from
     (same choice rule as convert_ct_hnscc.py), and the RTSTRUCT drawn on it
  2. pick the PRIMARY GTV by structure name:
       - only names containing "gtv"
       - excluded: lymph nodes ("node", "nodal", "ln") and treatment volumes
         ("ptv", "ctv", "nogtv", e.g. ptv70_nogtv)
       - post-chemo beats pre-chemo; an unlabeled GTV beats a pre-chemo one
       - a named primary ("gtv primary", "gtv bot", "gtv tonsil"...) beats plain "gtv",
         which beats any other gtv name (e.g. "gtv 70")
       - several structures tied at the top are merged into one mask
  3. fill the contours with rt-utils, line the slices up with the CT by their
     physical positions, and save masks/<id>.nii.gz with the CT's geometry
  4. write masks_report.csv: structure chosen, why, volume, number of separate
     pieces, mean CT value inside the mask, and flags to review

Run on the Windows PC from the LAMMF folder (REAL paths, not .lnk shortcuts):
    python convert_rtstruct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --limit 3
    python convert_rtstruct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --clinical "C:\\...\\clinical.xlsx"
(--clinical is optional: it adds each patient's N stage to the report.
 N0 = no lymph nodes involved, so a plain "gtv" must be the primary only.)

Needs convert_ct_hnscc.py in the same folder (reuses its CT-choice rule).
Printed output has counts only, no patient IDs. masks_report.csv has real IDs:
keep it on this PC (.gitignore excludes *.csv).
"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from rt_utils import RTStructBuilder

from convert_ct_hnscc import read_header, find_rt_study, pick_ct

PRIMARY_SITE_WORDS = {'primary', 'bot', 'tonsil', 'tongue', 'larynx', 'glottis',
                      'supraglottis', 'hypopharynx', 'oropharynx', 'nasopharynx'}


# ---------------------------------------------------------------- structure choice
def norm_name(name):
    """'GTV-Primary_70' -> 'gtv primary 70'"""
    return re.sub(r'[\s_\-/]+', ' ', str(name)).strip().lower()


def classify(name):
    """None if not a GTV; otherwise a dict describing it."""
    n = norm_name(name)
    words = n.split()
    compact = n.replace(' ', '')
    if 'gtv' not in compact:
        return None
    if 'nogtv' in compact or 'ptv' in compact or 'ctv' in compact:
        return {'excluded': 'treatment volume'}
    if 'node' in compact or 'nodal' in words or 'ln' in words:
        return {'excluded': 'lymph node'}
    return {'excluded': None,
            'pre': 'prechemo' in compact or 'prect' in compact or 'pre' in words,
            'post': 'postchemo' in compact or 'postct' in compact or 'post' in words,
            'plain': n == 'gtv',
            'named_primary': bool(PRIMARY_SITE_WORDS & set(words))}


def choose_structures(names):
    """Return (chosen names, reason, excluded gtv-like names)."""
    cands, excluded = [], []
    for nm in names:
        c = classify(nm)
        if c is None:
            continue
        if c['excluded']:
            excluded.append(f"{nm} ({c['excluded']})")
        else:
            cands.append((nm, c))
    if not cands:
        return [], 'no primary GTV', excluded

    notes = []
    post = [x for x in cands if x[1]['post']]
    pre = [x for x in cands if x[1]['pre']]
    if post:
        pool = post
        if pre:
            notes.append('post-chemo chosen over pre-chemo')
    else:
        unlabeled = [x for x in cands if not x[1]['pre']]
        if unlabeled:
            pool = unlabeled
            if pre:
                notes.append('pre-chemo ignored, unlabeled GTV used')
        else:
            pool = pre
            notes.append('ONLY PRE-CHEMO GTV')

    def rank(c):
        return 0 if c['named_primary'] else (1 if c['plain'] else 2)

    best = min(rank(c) for _, c in pool)
    chosen = [nm for nm, c in pool if rank(c) == best]
    reason = ['named primary', 'plain gtv', 'other gtv name'][best]
    if len(chosen) > 1:
        notes.append(f'{len(chosen)} structures merged')
    return chosen, '; '.join([reason] + notes), excluded


# ---------------------------------------------------------------- DICOM scanning
def scan_study(study_dir):
    """CT series [(dir, uid, n_files)], RTSTRUCTs [(path, referenced CT UIDs, roi names)]."""
    cts, rss = [], []
    for dirpath, _, files in os.walk(study_dir):
        dcms = sorted(f for f in files if f.lower().endswith('.dcm'))
        if not dcms:
            continue
        hdr = read_header(os.path.join(dirpath, dcms[0]))
        mod = str(getattr(hdr, 'Modality', '')) if hdr is not None else ''
        if mod == 'CT':
            cts.append((dirpath, str(getattr(hdr, 'SeriesInstanceUID', '')), len(dcms)))
        elif mod == 'RTSTRUCT':
            for f in dcms:
                rs = read_header(os.path.join(dirpath, f))
                if rs is None:
                    continue
                refs = set()
                for fr in getattr(rs, 'ReferencedFrameOfReferenceSequence', []):
                    for st in getattr(fr, 'RTReferencedStudySequence', []):
                        for se in getattr(st, 'RTReferencedSeriesSequence', []):
                            refs.add(str(se.SeriesInstanceUID))
                names = [str(r.ROIName).strip() for r in getattr(rs, 'StructureSetROISequence', [])]
                rss.append((os.path.join(dirpath, f), refs, names))
    return cts, rss


# ---------------------------------------------------------------- mask building
def load_rtstruct(series_dir, rs_path):
    try:
        return RTStructBuilder.create_from(dicom_series_path=series_dir,
                                           rt_struct_path=rs_path, warn_only=True)
    except TypeError:                       # older rt-utils without warn_only
        return RTStructBuilder.create_from(dicom_series_path=series_dir,
                                           rt_struct_path=rs_path)


def mask_on_ct_grid(rt, names, ct):
    """
    rt-utils returns (rows, cols, slices) in ITS slice order. Reorder to the CT's
    (slices, rows, cols) by matching each CT slice to the rt-utils slice at the
    same physical position, so nothing depends on how either library sorts files.
    """
    m = np.logical_or.reduce([rt.get_roi_mask_by_name(n) for n in names])
    direction = np.array(ct.GetDirection()).reshape(3, 3)
    normal = direction[:, 2]                                  # slice-stacking direction
    z_rt = np.array([np.dot(np.array(ds.ImagePositionPatient, float), normal)
                     for ds in rt.series_data])
    nz = ct.GetSize()[2]
    z_ct = np.array([np.dot(ct.TransformIndexToPhysicalPoint((0, 0, i)), normal)
                     for i in range(nz)])
    if m.shape[2] != len(z_rt) or m.shape[0] != ct.GetSize()[1] or m.shape[1] != ct.GetSize()[0]:
        raise RuntimeError(f'mask shape {m.shape} does not match CT size {ct.GetSize()}')
    tol = 0.5 * ct.GetSpacing()[2]
    out = np.zeros((nz, m.shape[0], m.shape[1]), dtype=np.uint8)
    for i, z in enumerate(z_ct):
        j = int(np.argmin(np.abs(z_rt - z)))
        if abs(z_rt[j] - z) > tol:
            raise RuntimeError(f'CT slice {i} has no RTSTRUCT image slice within {tol:.2f} mm')
        out[i] = m[:, :, j]
    return out


# ---------------------------------------------------------------- clinical N stage
def load_n_stage(path):
    if not path:
        return {}
    df = pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(path, encoding='latin-1')
    cols = {re.sub(r'\s+', ' ', str(c)).strip().lower(): c for c in df.columns}
    if 'tcia code' not in cols or 'n' not in cols:
        print("  (clinical file has no 'TCIA code' / 'N' column - N stage not added)")
        return {}
    df = df[df[cols['tcia code']].notna()]
    return dict(zip(df[cols['tcia code']].astype(str).str.strip(), df[cols['n']].astype(str).str.strip()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='REAL path to the folder of patient folders')
    ap.add_argument('--ct_dir', default='ct')
    ap.add_argument('--out', default='masks')
    ap.add_argument('--clinical', help='optional clinical Excel/CSV, to add N stage to the report')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    if a.root.lower().endswith('.lnk'):
        sys.exit("That's a .lnk shortcut. Use the shortcut's Target path instead.")
    if not os.path.isdir(a.ct_dir):
        sys.exit(f"No '{a.ct_dir}' folder here. Run convert_ct_hnscc.py first, from this folder.")
    os.makedirs(a.out, exist_ok=True)
    n_stage = load_n_stage(a.clinical)

    pids = sorted(f[:-len('.nii.gz')] for f in os.listdir(a.ct_dir) if f.endswith('.nii.gz'))
    if a.limit:
        pids = pids[:a.limit]
    print(f"patients with a CT in {a.ct_dir}/: {len(pids)}")

    rows = []
    for i, pid in enumerate(pids, 1):
        out_path = os.path.join(a.out, f'{pid}.nii.gz')
        row = {'Patient_ID': pid, 'status': '', 'chosen': '', 'reason': '', 'N_stage': n_stage.get(pid, ''),
               'excluded_gtv_like': '', 'volume_cc': '', 'n_pieces': '', 'mean_hu_in_mask': '', 'flags': ''}
        flags = []
        try:
            if os.path.exists(out_path) and not a.overwrite:
                row['status'] = 'already done'
                rows.append(row)
                print(f"[{i}/{len(pids)}] already done")
                continue
            studies = find_rt_study(os.path.join(a.root, pid))
            if not studies:
                raise RuntimeError('no RT Simulation study')
            cts, rss = scan_study(studies[0])
            all_refs = set().union(*[r[1] for r in rss]) if rss else set()
            (sdir, uid, _), _ = pick_ct(cts, all_refs)            # same rule as convert_ct_hnscc.py
            matching = [r for r in rss if uid in r[1]]
            if matching:
                rs_path, _, names = matching[0]
            elif len(rss) == 1:
                rs_path, _, names = rss[0]
                flags.append('RTSTRUCT does not reference this CT')
            else:
                raise RuntimeError('no RTSTRUCT for this CT')

            chosen, reason, excluded = choose_structures(names)
            row.update(chosen='; '.join(chosen), reason=reason, excluded_gtv_like='; '.join(excluded))
            if not chosen:
                row['status'] = 'skipped: no primary GTV'
            else:
                ct = sitk.ReadImage(os.path.join(a.ct_dir, f'{pid}.nii.gz'))
                rt = load_rtstruct(sdir, rs_path)
                mask = mask_on_ct_grid(rt, chosen, ct)
                n_vox = int(mask.sum())
                if n_vox == 0:
                    row['status'] = 'skipped: mask is empty'
                    flags.append('EMPTY')
                else:
                    img = sitk.GetImageFromArray(mask)
                    img.CopyInformation(ct)
                    sitk.WriteImage(img, out_path)
                    _, n_pieces = ndimage.label(mask, structure=np.ones((3, 3, 3)))
                    hu = sitk.GetArrayFromImage(ct)[mask.astype(bool)].mean()
                    row.update(status='converted', volume_cc=round(n_vox * np.prod(ct.GetSpacing()) / 1000, 1),
                               n_pieces=int(n_pieces), mean_hu_in_mask=round(float(hu), 1))
                    if n_pieces > 1 and 'plain gtv' in reason:
                        flags.append('PLAIN GTV IN SEVERAL PIECES (may include nodes)')
                    if hu < -300:
                        flags.append('LOW CT VALUE IN MASK (check alignment)')
                    if 'ONLY PRE-CHEMO' in reason:
                        flags.append('ONLY PRE-CHEMO GTV')
                    if 'merged' in reason:
                        flags.append('SEVERAL STRUCTURES MERGED')
        except Exception as e:
            row['status'] = f'ERROR: {type(e).__name__}: {e}'
        row['flags'] = '; '.join(flags)
        rows.append(row)
        msg = f"[{i}/{len(pids)}] {row['status']}"
        if row['status'] == 'converted':
            msg += f"  ({row['reason']}, {row['volume_cc']} cc, {row['n_pieces']} piece(s))"
        if flags:
            msg += f"  FLAG: {row['flags']}"
        print(msg)

    rep = pd.DataFrame(rows)
    rep.to_csv('masks_report.csv', index=False)
    print("\nsummary:")
    print(rep['status'].str.replace(r':.*', '', regex=True).value_counts().to_string())
    done = rep[rep['status'] == 'converted']
    if len(done):
        print("\nhow the structure was chosen:")
        print(done['reason'].value_counts().to_string())
        vol = pd.to_numeric(done['volume_cc'])
        print(f"\ntumor volume (cc): min {vol.min()} / median {vol.median()} / max {vol.max()}")
    flagged = rep[rep['flags'] != '']
    print(f"\npatients flagged for review: {len(flagged)}")
    for f, n in flagged['flags'].str.split('; ').explode().value_counts().items():
        print(f"  {n:3d}  {f}")
    print(f"\nmasks in: {a.out}/   details: masks_report.csv (real IDs, keep on this PC)")


if __name__ == '__main__':
    main()
