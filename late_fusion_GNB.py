"""
Late-fusion script — plumbing fixes only, fusion logic unchanged.
 
WHAT CHANGED (and nothing else):
  - import models from local Model_.py       (was: from GTC_code.Model_ ...)
  - import dataset as NPC_OS_Dataset          (matches your renamed class)
  - TEXT branch removed everywhere            (you didn't train it)
  - weight paths -> best/best_model_gnn.pth, best/best_model_cnn.pth
  - CNN built with input_size=(16,32,32)      (matches how you trained it)
  - data moved to device in every loop        (device-agnostic; runs on your Mac)
  - dropped dead code that referenced text / patient_id / val (unused in output)
 
The Gaussian Naive Bayes fusion, the probability collection, and all the
reported metrics are exactly as the authors wrote them (minus the text column).
"""
 
import os
import joblib
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
 
from Model_ import Model_CNN, Model_GNN          # CHANGED: local import, no text
from data_utils import NPC_OS_Dataset            # CHANGED: your renamed class
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
dataset_path = 'dataset'
csv_dir = 'csv_data'
all_data_path = 'csv_data/all'
 
train_dataset = NPC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir,
                               set_name='train', all_data_path=all_data_path)
test_dataset = NPC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir,
                              set_name='test', all_data_path=all_data_path)
 
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
 
# ---- build models exactly as trained, load saved weights ----
model_GNN = Model_GNN(177, 1).to(device)
model_GNN.load_state_dict(torch.load('best/best_model_gnn.pth', weights_only=False))
 
# CHANGED: (16,32,32) to match your shrunk training config, or the weights won't load
model_CNN = Model_CNN(1, 128, (16, 32, 32)).to(device)
model_CNN.load_state_dict(torch.load('best/best_model_cnn.pth', weights_only=False))
 
model_GNN.eval()
model_CNN.eval()
 
# ---- collect TRAIN branch probabilities ----
train_cnn_probs, train_gnn_probs, train_targets = [], [], []
with torch.no_grad():
    for data in train_loader:
        data = data.to(device)
        cnn_p, _ = model_CNN(data)
        gnn_p, _ = model_GNN(data)
        train_cnn_probs.append(cnn_p.cpu().numpy())
        train_gnn_probs.append(gnn_p.cpu().numpy())
        train_targets.append(data.y.cpu().numpy())
train_cnn_probs = np.vstack(train_cnn_probs)
train_gnn_probs = np.vstack(train_gnn_probs)
train_targets = np.hstack(train_targets)
 
# ---- collect TEST branch probabilities ----
test_cnn_probs, test_gnn_probs, test_targets = [], [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        cnn_p, _ = model_CNN(data)
        gnn_p, _ = model_GNN(data)
        test_cnn_probs.append(cnn_p.cpu().numpy())
        test_gnn_probs.append(gnn_p.cpu().numpy())
        test_targets.append(data.y.cpu().numpy())
test_cnn_probs = np.vstack(test_cnn_probs)
test_gnn_probs = np.vstack(test_gnn_probs)
test_targets = np.hstack(test_targets)
 
# ---- Gaussian Naive Bayes fusion (unchanged, minus the text column) ----
lr = GaussianNB()
train_fusion_features = np.column_stack([train_cnn_probs, train_gnn_probs])
test_fusion_features = np.column_stack([test_cnn_probs, test_gnn_probs])
lr.fit(train_fusion_features, train_targets)
 
os.makedirs('best', exist_ok=True)
joblib.dump(lr, 'best/fusion_gnb_model.joblib')
 
nb_train_pred = lr.predict(train_fusion_features)
nb_test_pred = lr.predict(test_fusion_features)
 
# per-branch probability for AUC (single sigmoid output -> flatten)
train_cnn_auc = train_cnn_probs.flatten()
train_gnn_auc = train_gnn_probs.flatten()
test_cnn_auc = test_cnn_probs.flatten()
test_gnn_auc = test_gnn_probs.flatten()
 
nb_train_probs = lr.predict_proba(train_fusion_features)
nb_test_probs = lr.predict_proba(test_fusion_features)
if nb_train_probs.shape[1] == 2:
    nb_train_probs = nb_train_probs[:, 1]
    nb_test_probs = nb_test_probs[:, 1]
else:
    nb_train_probs = nb_train_probs.flatten()
    nb_test_probs = nb_test_probs.flatten()
 
 
def safe_auc(y, p):
    # AUC is undefined if a split is single-class (common with tiny test sets)
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float('nan')
 
 
# NOTE: the per-branch "Accuracy" below uses argmax(axis=1) exactly as the
# authors wrote it. Because these models output a single sigmoid value (not two
# class columns), argmax is always 0 -> that accuracy is degenerate. Trust the
# AUC and the NB-fusion numbers. Left as-is to avoid changing the original logic.
print("=== Single Model Performance ===")
print(f"CNN Train Acc: {accuracy_score(train_targets, train_cnn_probs.argmax(axis=1)):.4f}")
print(f"CNN Test Acc:  {accuracy_score(test_targets, test_cnn_probs.argmax(axis=1)):.4f}")
print(f"CNN Train AUC: {safe_auc(train_targets, train_cnn_auc):.4f}")
print(f"CNN Test AUC:  {safe_auc(test_targets, test_cnn_auc):.4f}")
print('=' * 50)
print(f"GNN Train Acc: {accuracy_score(train_targets, train_gnn_probs.argmax(axis=1)):.4f}")
print(f"GNN Test Acc:  {accuracy_score(test_targets, test_gnn_probs.argmax(axis=1)):.4f}")
print(f"GNN Train AUC: {safe_auc(train_targets, train_gnn_auc):.4f}")
print(f"GNN Test AUC:  {safe_auc(test_targets, test_gnn_auc):.4f}")
 
print("\n=== Fusion Model Performance ===")
print(f"Naive Bayes Fusion - Train Acc: {accuracy_score(train_targets, nb_train_pred):.4f}")
print(f"Naive Bayes Fusion - Test Acc:  {accuracy_score(test_targets, nb_test_pred):.4f}")
print(f"Naive Bayes Fusion - Train AUC: {safe_auc(train_targets, nb_train_probs):.4f}")
print(f"Naive Bayes Fusion - Test AUC:  {safe_auc(test_targets, nb_test_probs):.4f}")
 
print("\n=== Best Fusion Strategy Detailed Report ===")
print(classification_report(test_targets, nb_test_pred))