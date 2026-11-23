# WaferSense AI - Explainable Wafer Defect Classification (SIC Capstone)

Software-only prototype: CNN classifies WM-811K wafer maps into 9 pattern classes, Grad-CAM shows which regions
influenced the prediction, Streamlit dashboard presents everything.

## Files
| File | Purpose |
|---|---|
| `common.py` | preprocessing, CNN, Grad-CAM, WM-811K loader, synthetic data (test only) |
| `train.py` | EDA counts -> subset -> stratified split -> RF baseline -> CNN (class-weighted) -> test evaluation |
| `app.py` | Streamlit dashboard: Home, Wafer Inspector, Explainability, Analytics, Model Performance, Report Export |
| `WaferSense_Colab.ipynb` | one-click training on free Colab GPU |

## Run (recommended: train on Colab, run app locally)
1. Open `WaferSense_Colab.ipynb` in Google Colab (Runtime -> GPU), run all cells. It downloads WM-811K, trains,
   and gives you `artifacts.zip`.
2. On your laptop:
```bash
pip install -r requirements.txt
# unzip artifacts.zip here, so the folder ./artifacts exists
streamlit run app.py
```

## Train locally instead
```bash
python train.py --data /path/to/folder_with_LSWMD.pkl --epochs 20 --max_per_class 4000
```
Quick code test without the dataset: `python train.py --synthetic --epochs 3` (fake data, numbers meaningless).

## Design notes (match the proposal)
- Maps encoded 0 = background, 1 = pass die, 2 = fail die; resized to 64x64 with nearest-neighbour; one-hot 3 channels.
- Only labelled records are used; `--max_per_class` caps the huge "none" class (documented subset).
- Stratified 70/15/15 split; test set evaluated once at the end.
- Imbalance: class-weighted cross-entropy; augmentation limited to rotation/flip (label-preserving).
- Metrics: accuracy, macro-F1, balanced accuracy, per-class P/R/F1, confusion matrix, RF baseline comparison.
- Grad-CAM on the last conv block. It is an inspection aid, not proof of causality.
- Limitation: split is stratified by label, not grouped by lot; mention this in your report (proposal's group-aware split is "where metadata permits").
- Cite the WM-811K source (Wu et al., IEEE TSM 2015) and check the dataset licence on Kaggle.
