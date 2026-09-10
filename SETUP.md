# Setup Guide — installing the requirements

Follow these steps in order. Works on Windows and Mac (commands are the same
under conda). Copy-paste one block at a time.

---

## 1. Install Miniconda (skip if you already have Anaconda/conda)

Download and install Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Windows: run the .exe installer.
- Mac: run the .pkg installer.

After installing, open a **new** terminal (Windows: "Anaconda Prompt"; Mac: Terminal)
and confirm it works:

```
conda --version
```

---

## 2. Create a fresh environment

Keeps this project's packages isolated from everything else.

```
conda create -n lammf python=3.12 -y
conda activate lammf
```

You should now see `(lammf)` at the start of your prompt. Do all the next steps
with this environment active.

---

## 3. Install PyTorch FIRST (choose GPU or CPU)

PyTorch must be installed on its own, because the right command depends on your
hardware. Go to the official selector and pick your options:

    https://pytorch.org/get-started/locally/

- **If your machine has an NVIDIA GPU** → choose "CUDA" (training will be far
  faster; needed for full-resolution runs).
- **If not** (or you're unsure) → choose "CPU".

The site gives you an exact command. Examples (yours may differ — use the site's):

```
# NVIDIA GPU example (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only example:
pip install torch
```

Verify it installed and whether it sees a GPU:

```
python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
```

`GPU available: True` means CUDA is working. `False` = CPU mode (fine, just slower).

---

## 4. Install everything else

With `torch` already installed, install the rest (this also adds
torch_geometric, which needs torch present first):

```
pip install -r requirements.txt
```

(pip will skip torch since it's already installed from step 3.)

---

## 5. Verify the whole stack imports

```
python -c "import torch, torch_geometric, mirp, SimpleITK, pydicom, pandas, sklearn; print('ALL IMPORTS OK | pydicom', pydicom.__version__)"
```

You want `ALL IMPORTS OK` and `pydicom 2.4.4`. If pydicom prints a 3.x version,
fix it with:

```
pip install "pydicom==2.4.4"
```

(pydicom 3.x breaks the mask converter — this pin matters.)

---

## Done

The environment is ready. See the README / PROGRESS_SUMMARY for how to download
the data and run the pipeline (make_labels -> convert_dicom -> extract_features_mirp
-> build_real_csvs -> trainer -> late_fusion_GNB).

## Troubleshooting
- "No module named X" -> make sure `(lammf)` is active and rerun step 4.
- torch_geometric errors -> confirm torch installed cleanly in step 3 first.
- pydicom-seg / mask errors -> confirm `pydicom==2.4.4` (step 5).
- Fresh terminal doesn't show conda -> reopen "Anaconda Prompt" (Windows).
