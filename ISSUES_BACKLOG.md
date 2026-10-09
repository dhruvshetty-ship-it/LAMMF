# LAMMF — Issues Backlog (make-it-right phase)

Every issue flagged so far, compressed. Pipeline WORKS end-to-end; these are
improvements. ★ = highest value. Tackle data first (per plan).

## DATA PROCESSING
- [x] ★ **D1 — Radiomics not normalized.** DONE: z-score via StandardScaler,
      fit on TRAIN only, transform val/test with train stats. Only the 177 feature
      columns scaled. (Scaler not yet persisted with joblib — optional, for cluster.)
- [x] ★ **D2 — Split is random, not stratified.** DONE: two-step stratified
      train_test_split (peel off 30%, then split that 50/50 into val/test), both
      divisions stratified on labels. 70/15/15, class ratio preserved.
- [ ] **D3 — CT selection = "biggest series" (crude).** Should use the CT the
      DICOM-SEG *references* to guarantee mask/CT alignment. Correctness risk at scale.
- [ ] **D4 — Empty masks** (e.g. R01-022 dropped). Find WHY (label value? geometry?)
      before scaling to 144, or lose usable patients silently.
- [ ] **D5 — Trivial GNN graph.** build_real_csvs gives 1 ROI/patient -> single-node
      graph. Design call: MLP, tabular model, or build multi-node graphs. (see M4)
- [ ] **D6 — MIRP settings are defaults** (fixed_bin 32, 2mm resample, base features
      only, no wavelet/LoG). Confirm choices with advisor; consider fuller feature set.
- [ ] **D7 — Cosmetic:** delete debug print loop in data_utils.py (~lines 63-68).

## MODEL ARCHITECTURE
- [ ] ★ **M1 — CNN loads+preprocesses images INSIDE forward(), every epoch.** Brutally
      slow + wrong separation + blocks augmentation. Fix: move to Dataset, cache once.
- [ ] ★ **M2 — Overfitting: huge FC layer + no dropout on ~27 train patients.** CNN has
      millions of params. Fix: dropout, weight decay, smaller model, more data.
- [ ] **M3 — BatchNorm with batch_size=3 is unreliable** (noisy stats, size-1 crash).
      Fix: GroupNorm / InstanceNorm (batch-independent).
- [ ] **M4 — GNN runs on single-node graphs** -> message-passing/TopKPooling do nothing.
      Simplify to MLP or construct real multi-node graphs. (paired with D5)
- [ ] **M5 — CNN crop = (16,32,32)** (smoke-test size, throws away the scan). Bump toward
      (64,224,224) for real training. NEEDS GPU.
- [ ] **M6 — No data augmentation.** Small N -> overfitting. Add 3D augmentations.

## TRAINER
- [x] ★ **T1 — Unweighted BCELoss -> model collapses to "survived."** DONE: per-sample
      weighted BCE via F.binary_cross_entropy. Two per-class weights (W_DIED~2.9,
      W_SURVIVED~0.6, inverse-frequency) stamped onto each batch with torch.where.
      Kept sigmoid + BCELoss route (no logits switch). Validation stays unweighted.
- [ ] **T2 — Model selection / early stop on 5-patient val LOSS** (noisy; also monitors
      loss not AUC). Fix: monitor AUC; ultimately needs bigger val / cross-validation.
- [x] ★ **T3 — No random seeds** -> non-reproducible. DONE: np.random.seed(42) +
      torch.manual_seed(42) at top of trainer (before model build). random.seed not
      needed (trainer doesn't use it). cuDNN determinism deferred to GPU/cluster.
- [ ] **T4 — No LR schedule; LR possibly high (0.005).** Add decay / tune LR.
- [ ] **T5 — Per-branch accuracy uses argmax on single-output** -> always predicts 0
      (degenerate). Fix: threshold at 0.5. (reporting only; trust AUC/fusion)
- [ ] **T6 — Verify the test_auc print fix** is fully in (unpack `test_loss, test_auc`).

## DEFERRED / SCOPE (bigger, advisor/compute-gated)
- [ ] **S1 — Censoring:** 148 "alive" labeled survived without follow-up check. Refine.
- [ ] **S2 — Cutoff:** 2yr vs 5yr (changes labels + balance). Advisor decision.
- [ ] **S3 — GPU** needed for full-res CNN + full cohort (laptop can't).
- [ ] **S4 — Head/neck -> lung:** training OUR model, not reproducing the paper.
- [ ] **S5 — Scale 40 -> full ~144 masked cohort.**
- [ ] **S6 — No cross-validation** (single split = high variance at N=39).

## Suggested order
1. ~~D1 + D2 (normalize + stratify)~~ — DONE.
2. ~~T3 (seeds)~~ — DONE (np + torch seeded in trainer).
3. ~~T1 (weighted loss)~~ — DONE.
4. M1 (preprocess out of forward) — makes CNN iteration fast + enables aug. **← NEXT**
5. M2/M3/M6 (regularization, norm, augmentation) — attack overfitting.
6. D5/M4 (GNN-is-an-MLP) — design decision.
7. Scale-up (S3/S5) once the above are solid.
