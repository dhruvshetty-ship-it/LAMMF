"""
Find your USABLE cohort = patients that have BOTH a survival label AND imaging.
You can only train on the intersection (need the scan AND the outcome).

Pulls the imaging patient list straight from TCIA's API (NO image download
needed), intersects it with your labels file, writes cohort.csv, and prints the
real N you'd actually train on.

Usage:
    python make_cohort.py labels_all.csv                       # fetches list live
    python make_cohort.py labels_all.csv --image_ids_file ids.txt   # offline

(labels_all.csv comes from make_labels.py: columns Patient_ID,label)
"""

import argparse
import json
import urllib.request
import urllib.parse
import pandas as pd

COLLECTION = "NSCLC Radiogenomics"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getPatient"


def get_imaging_ids(image_ids_file=None):
    if image_ids_file:                                   # offline: one ID per line
        with open(image_ids_file) as f:
            return {line.strip() for line in f if line.strip()}
    url = API + "?" + urllib.parse.urlencode({"Collection": COLLECTION})
    with urllib.request.urlopen(url) as r:               # live: query TCIA
        data = json.load(r)
    return {rec["PatientId"] for rec in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv")
    ap.add_argument("--image_ids_file", default=None)
    ap.add_argument("--out", default="cohort.csv")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels_csv)
    label_ids = set(labels["Patient_ID"].astype(str))
    image_ids = get_imaging_ids(a.image_ids_file)

    both = label_ids & image_ids                         # the usable cohort
    labels_only = label_ids - image_ids
    images_only = image_ids - label_ids

    cohort = labels[labels["Patient_ID"].astype(str).isin(both)].copy()
    cohort.to_csv(a.out, index=False)

    print(f"patients with a label:          {len(label_ids)}")
    print(f"patients with imaging:          {len(image_ids)}")
    print(f"USABLE (have BOTH) -> real N:   {len(both)}")
    print(f"  dropped (label but no image): {len(labels_only)}")
    print(f"  dropped (image but no label): {len(images_only)}")
    if both:
        pos = int((cohort["label"] == 1).sum())
        neg = int((cohort["label"] == 0).sum())
        print(f"cohort balance: {pos} survived (1) / {neg} died (0)")
    # sanity check that IDs are the same format on both sides
    print("sample label IDs:", sorted(label_ids)[:3])
    print("sample image IDs:", sorted(image_ids)[:3])
    print(f"wrote {len(cohort)} rows -> {a.out}")


if __name__ == "__main__":
    main()
