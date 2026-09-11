"""
Generate a tiny FAKE dataset to smoke-test the pipeline on any machine (no
download needed). Verifies MIRP, build_real_csvs, the trainer, and fusion all
run — good for confirming a fresh (e.g. Windows) environment works.

Creates:
  ct/<id>.nii.gz     - fake CT volumes (noise + a bright "tumor" cube)
  masks/<id>.nii.gz  - matching binary tumor masks
  labels_all.csv     - Patient_ID,label   (label correlates with tumor intensity,
                       so the models can learn something above chance)

Then run the pipeline (skips convert_dicom, which needs real DICOM):
  python extract_features_mirp.py
  python build_real_csvs.py labels_all.csv
  python trainer.py gnn
  python trainer.py cnn
  python late_fusion_GNB.py

Uses SimpleITK (already installed). Small volumes -> fast on CPU.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

N = 20
VOL = (40, 80, 80)          # (z, y, x)
RNG = np.random.default_rng(0)


def save(arr, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    img = sitk.GetImageFromArray(arr)      # preserves dtype
    img.SetSpacing((1.0, 1.0, 2.0))        # (x, y, z) mm
    sitk.WriteImage(img, path)


def main():
    rows = []
    for i in range(N):
        pid = f"R01-{i:03d}"
        sign = 1 if RNG.random() > 0.5 else -1
        ct = RNG.normal(-500, 50, VOL).astype(np.float32)   # noisy background
        mask = np.zeros(VOL, np.uint8)
        dz, dy, dx = 8, 16, 16
        z0 = int(RNG.integers(0, VOL[0] - dz))
        y0 = int(RNG.integers(0, VOL[1] - dy))
        x0 = int(RNG.integers(0, VOL[2] - dx))
        intensity = 300 + 250 * sign + RNG.normal(0, 40)     # class-dependent brightness
        ct[z0:z0+dz, y0:y0+dy, x0:x0+dx] += intensity
        mask[z0:z0+dz, y0:y0+dy, x0:x0+dx] = 1

        save(ct, os.path.join("ct", f"{pid}.nii.gz"))
        save(mask, os.path.join("masks", f"{pid}.nii.gz"))
        rows.append({"Patient_ID": pid, "label": 1 if sign > 0 else 0})

    pd.DataFrame(rows).to_csv("labels_all.csv", index=False)
    pos = sum(r["label"] for r in rows)
    print(f"wrote {N} fake patients -> ct/, masks/, labels_all.csv "
          f"({pos} survived / {N-pos} died)")


if __name__ == "__main__":
    main()
