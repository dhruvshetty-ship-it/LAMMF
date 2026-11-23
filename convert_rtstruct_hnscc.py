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
  3. fill the contours directly on the saved CT's grid (each contour point is a
     real-world position in mm; the CT's geometry turns it into a voxel), and
     save masks/<id>.nii.gz with the CT's geometry
  4. write masks_report.csv: structure chosen, why, volume, number of separate
     pieces, mean CT value inside the mask, and flags to review

Run on the Windows PC from the LAMMF folder (REAL paths, not .lnk shortcuts):
    python convert_rtstruct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --limit 3
    python convert_rtstruct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --clinical "C:\\...\\clinical.xlsx"
(--clinical is optional: it adds each patient's N stage to the report.
 N0 = no lymph nodes involved, so a plain "gtv" must be the primary only.)

If some patients error, fix the cause, then re-run only those:
    python convert_rtstruct_hnscc.py "V:\\QIXR\\TCIA_H&N\\HNSCC" --retry_errors --clinical "..."

Needs convert_ct_hnscc.py in the same folder (reuses its CT-choice rule).
Fills the contours itself (pixel counts as tumor if its center is inside the
outline), so rt-utils / OpenCV aren't needed.
Printed output has counts only, no patient IDs. masks_report.csv has real IDs:
keep it on this PC (.gitignore excludes *.csv).
"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage

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
def contour_points(rs, names):
    """All contour polygons (each an N x 3 array of x,y,z in mm) for the chosen structures."""
    number_of = {str(r.ROIName).strip(): int(r.ROINumber)
                 for r in getattr(rs, 'StructureSetROISequence', [])}
    wanted = {number_of[n] for n in names if n in number_of}
    polys = []
    for rc in getattr(rs, 'ROIContourSequence', []):
        if int(getattr(rc, 'ReferencedROINumber', -1)) not in wanted:
            continue
        for c in getattr(rc, 'ContourSequence', []):
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            if len(pts) >= 3:                       # skip single points / lines
                polys.append(pts)
    return polys


def fill_polygon(ny, nx, cols, rows):
    """
    Pixels whose CENTERS lie inside the polygon (even-odd rule), row by row.
    cols/rows are the vertices in voxel units; pixel (r, c) has its center at (c, r).
    """
    layer = np.zeros((ny, nx), dtype=np.uint8)
    c0, r0 = cols, rows
    c1, r1 = np.roll(cols, -1), np.roll(rows, -1)
    for r in range(max(int(np.ceil(rows.min())), 0), min(int(np.floor(rows.max())), ny - 1) + 1):
        crosses = (r0 > r) != (r1 > r)                 # edges that cross this row's center line
        if not crosses.any():
            continue
        xs = np.sort(c0[crosses] + (r - r0[crosses]) * (c1[crosses] - c0[crosses]) / (r1[crosses] - r0[crosses]))
        for a, b in zip(xs[0::2], xs[1::2]):           # inside between each pair of crossings
            lo, hi = max(int(np.ceil(a)), 0), min(int(np.floor(b)), nx - 1)
            if lo <= hi:
                layer[r, lo:hi + 1] = 1
    return layer


def mask_on_ct_grid(rs, names, ct):
    """
    Fill the contours directly on the saved CT's voxel grid.
    Each contour point is a real-world position (mm). The CT's own geometry
    (origin, spacing, direction) converts it to a voxel position, so the mask
    can only line up with the CT it will be trained with. Polygons on the same
    slice are combined with XOR, so a contour inside another one makes a hole.
    """
    polys = contour_points(rs, names)
    if not polys:
        raise RuntimeError('the chosen structure has no contour points')
    nx, ny, nz = ct.GetSize()
    origin = np.array(ct.GetOrigin())
    to_mm = np.array(ct.GetDirection()).reshape(3, 3) @ np.diag(ct.GetSpacing())
    to_voxel = np.linalg.inv(to_mm)
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    for pts in polys:
        idx = (pts - origin) @ to_voxel.T          # columns: column, row, slice (voxel units)
        k = float(idx[:, 2].mean())
        kk = int(round(k))
        if abs(k - kk) > 0.3 or not 0 <= kk < nz:
            raise RuntimeError(f'contour lands at slice position {k:.2f}, not on a CT slice '
                               f'(CT has {nz} slices of {ct.GetSpacing()[2]:.2f} mm)')
        mask[kk] ^= fill_polygon(ny, nx, idx[:, 0], idx[:, 1])
    return mask


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
    ap.add_argument('--retry_errors', action='store_true',
                    help='only re-run patients whose status was ERROR in the last masks_report.csv')
    a = ap.parse_args()

    if a.root.lower().endswith('.lnk'):
        sys.exit("That's a .lnk shortcut. Use the shortcut's Target path instead.")
    if not os.path.isdir(a.ct_dir):
        sys.exit(f"No '{a.ct_dir}' folder here. Run convert_ct_hnscc.py first, from this folder.")
    os.makedirs(a.out, exist_ok=True)
    n_stage = load_n_stage(a.clinical)

    pids = sorted(f[:-len('.nii.gz')] for f in os.listdir(a.ct_dir) if f.endswith('.nii.gz'))
    previous = []
    if a.retry_errors:
        if not os.path.exists('masks_report.csv'):
            sys.exit('--retry_errors needs a masks_report.csv from an earlier run')
        old = pd.read_csv('masks_report.csv', dtype=str).fillna('')
        retry = set(old.loc[old['status'].str.startswith('ERROR'), 'Patient_ID'])
        previous = [r for _, r in old.iterrows() if r['Patient_ID'] not in retry]
        previous = [dict(r) for r in previous]
        pids = [p for p in pids if p in retry]
        print(f"retrying {len(pids)} patient(s) that errored last time")
    if a.limit:
        pids = pids[:a.limit]
    print(f"patients with a CT in {a.ct_dir}/ to process: {len(pids)}")

    rows = []
    for i, pid in enumerate(pids, 1):
        out_path = os.path.join(a.out, f'{pid}.nii.gz')
        row = {'Patient_ID': pid, 'status': '', 'chosen': '', 'reason': '', 'N_stage': n_stage.get(pid, ''),
               'excluded_gtv_like': '', 'volume_cc': '', 'n_pieces': '', 'piece_sizes_cc': '',
               'largest_piece_pct': '', 'mean_hu_in_mask': '', 'flags': ''}
        flags = []
        try:
            if os.path.exists(out_path) and not (a.overwrite or a.retry_errors):
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

            rs = pydicom.dcmread(rs_path, force=True)
            drawn = [n for n in names if contour_points(rs, [n])]   # ignore empty placeholder structures
            empty = [f'{n} (empty)' for n in names if n not in drawn and classify(n) is not None]
            chosen, reason, excluded = choose_structures(drawn)
            row.update(chosen='; '.join(chosen), reason=reason, excluded_gtv_like='; '.join(excluded + empty))
            if not chosen:
                row['status'] = 'skipped: no primary GTV'
            else:
                ct = sitk.ReadImage(os.path.join(a.ct_dir, f'{pid}.nii.gz'))
                mask = mask_on_ct_grid(rs, chosen, ct)
                n_vox = int(mask.sum())
                if n_vox == 0:
                    row['status'] = 'skipped: mask is empty'
                    flags.append('EMPTY')
                else:
                    img = sitk.GetImageFromArray(mask)
                    img.CopyInformation(ct)
                    sitk.WriteImage(img, out_path)
                    labeled, n_pieces = ndimage.label(mask, structure=np.ones((3, 3, 3)))
                    voxel_cc = np.prod(ct.GetSpacing()) / 1000
                    sizes = np.sort(np.bincount(labeled.ravel())[1:])[::-1] * voxel_cc   # cc, largest first
                    shown = ', '.join(f'{s:.2f}' for s in sizes[:6])
                    if n_pieces > 6:
                        shown += f' (+{n_pieces - 6} more)'
                    hu = sitk.GetArrayFromImage(ct)[mask.astype(bool)].mean()
                    row.update(status='converted', volume_cc=round(n_vox * voxel_cc, 1),
                               n_pieces=int(n_pieces), piece_sizes_cc=shown,
                               largest_piece_pct=round(100 * sizes[0] / sizes.sum(), 1),
                               mean_hu_in_mask=round(float(hu), 1))
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
        if not row['status'].startswith(('converted', 'already done')) and os.path.exists(out_path):
            os.remove(out_path)          # don't leave a stale mask from an earlier run
        row['flags'] = '; '.join(flags)
        rows.append(row)
        msg = f"[{i}/{len(pids)}] {row['status']}"
        if row['status'] == 'converted':
            msg += f"  ({row['reason']}, {row['volume_cc']} cc, {row['n_pieces']} piece(s)"
            if row['n_pieces'] > 1:
                msg += f": {row['piece_sizes_cc']} cc, largest = {row['largest_piece_pct']}%"
            msg += ", mean CT " + f"{row['mean_hu_in_mask']} HU)"
        if flags:
            msg += f"  FLAG: {row['flags']}"
        print(msg)

    rep = pd.DataFrame(previous + rows)          # with --retry_errors, keeps the earlier successes
    rep.to_csv('masks_report.csv', index=False)
    rep = rep.fillna('')
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
