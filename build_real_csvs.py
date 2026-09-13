"""
Assemble the model-ready CSVs from your converted real data (CNN-first run).

Reads:
  - labels_all.csv  (Patient_ID,label  -- from make_labels.py)
  - the ct/ and masks/ folders produced by convert_dicom.py
Keeps only patients that have CT + mask + label, splits train/val/test, and
writes every file your data_utils.py reads:

  normalized_data_csv/normalized_data_{train,validation,test}.csv
  train_data_csv/{train,validation,test}_set.csv
  csv_data/file_paths_CT.csv
  all_clin_data.csv                       (placeholder, keyed to real IDs)
  Text_data/137_clinical_descriptions.csv (placeholder) + 3 empty siblings

The radiomics feature columns are PLACEHOLDER ZEROS. That's fine for the CNN
branch (it ignores x). When you add the GNN, replace normalized_data_* with real
PyRadiomics output.

Run from your LAMMF folder (where ct/ and masks/ live):
    python build_real_csvs.py labels_all.csv
"""

import os
import argparse
import random
import pandas as pd

N_DUMMY_FEAT = 1302   # match the GNN's expected input_dim (paper used 1302)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels_csv')
    ap.add_argument('--ct_dir', default='ct')
    ap.add_argument('--mask_dir', default='masks')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    labels = pd.read_csv(a.labels_csv)
    label_map = {str(p): int(l) for p, l in zip(labels['Patient_ID'], labels['label'])}

    # Use REAL features from extract_features.py if present; else random placeholders.
    feat_df, feat_cols = None, None
    if os.path.exists('radiomics_features.csv'):
        feat_df = pd.read_csv('radiomics_features.csv')
        feat_df['Patient_ID'] = feat_df['Patient_ID'].astype(str)
        feat_df = feat_df.set_index('Patient_ID')
        feat_cols = list(feat_df.columns)
        print(f"using REAL features from radiomics_features.csv: {len(feat_cols)} per patient")
    else:
        print(f"no radiomics_features.csv -> using {N_DUMMY_FEAT} RANDOM placeholder features")

    cohort = [pid for pid in label_map
              if os.path.exists(os.path.join(a.ct_dir, f'{pid}.nii.gz'))
              and os.path.exists(os.path.join(a.mask_dir, f'{pid}.nii.gz'))]
    if feat_df is not None:                       # keep only patients that also have features
        cohort = [pid for pid in cohort if pid in feat_df.index]
    cohort.sort()
    if not cohort:
        print("No patient has BOTH ct/<id>.nii.gz and masks/<id>.nii.gz + a label.")
        print("Check convert_dicom.py finished and you're running in the right folder.")
        return
    print(f"trainable cohort (CT + mask + label): {len(cohort)}")

    random.seed(a.seed)
    random.shuffle(cohort)
    n = len(cohort)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    splits = {
        'train': cohort[:n_tr],
        'validation': cohort[n_tr:n_tr + n_va],
        'test': cohort[n_tr + n_va:],
    }

    for d in ['normalized_data_csv', 'train_data_csv', 'csv_data', 'Text_data']:
        os.makedirs(d, exist_ok=True)

    all_paths, all_clin, all_text = [], [], []
    for split, pids in splits.items():
        feat_rows, label_rows = [], []
        for pid in pids:
            # column order MUST be: Patient_ID, Component_ID, ROI_Name, Mask_Path, <features...>
            row = {'Patient_ID': pid, 'Component_ID': 'C0', 'ROI_Name': 'tumor',
                   'Mask_Path': os.path.join(a.mask_dir, f'{pid}.nii.gz')}
            if feat_df is not None:               # real features from extract_features.py
                for c in feat_cols:
                    row[c] = float(feat_df.loc[pid, c])
            else:                                 # random stand-in
                for f in range(N_DUMMY_FEAT):
                    row[f'feat_{f}'] = random.gauss(0, 1)
            feat_rows.append(row)
            label_rows.append({'Patient_ID': pid, 'label': label_map[pid]})
            all_paths.append({'Patient_ID': pid,
                              'CT_Path': os.path.join(a.ct_dir, f'{pid}.nii.gz'),
                              'Mask_Path': os.path.join(a.mask_dir, f'{pid}.nii.gz')})
            all_clin.append({'Patient_ID': pid, **{f'clin{i}': 0.0 for i in range(8)}})
            all_text.append({'id': pid, 'description': 'na'})
        pd.DataFrame(feat_rows).to_csv(
            f'normalized_data_csv/normalized_data_{split}.csv', index=False)
        pd.DataFrame(label_rows).to_csv(
            f'train_data_csv/{split}_set.csv', index=False)
        pos = sum(r['label'] for r in label_rows)
        print(f"  {split:11s}: {len(pids):3d} patients ({pos} survived / {len(pids)-pos} died)")
    
    # normalized_train_df = pd.read_csv('normalized_data_train.csv')
    # normalized_test_df = 
    # normalized_val_df =  



    pd.DataFrame(all_paths).to_csv('csv_data/file_paths_CT.csv', index=False)
    pd.DataFrame(all_clin).to_csv('all_clin_data.csv', index=False)
    pd.DataFrame(all_text).to_csv('Text_data/137_clinical_descriptions.csv', index=False)
    for name in ['298', '495', '606']:
        pd.DataFrame(columns=['id', 'description']).to_csv(
            f'Text_data/{name}_clinical_descriptions.csv', index=False)

    n_feat = len(feat_cols) if feat_cols else N_DUMMY_FEAT
    print("wrote all CSVs.")
    print(f">>> GNN input_dim must be {n_feat}: set Model_GNN(input_dim={n_feat}, ...) "
          f"in trainer.py AND late_fusion_GNB.py")
    print("IMPORTANT: delete any old 'dataset/' cache folder before training so it "
          "rebuilds from these new files.")


if __name__ == '__main__':
    main()
