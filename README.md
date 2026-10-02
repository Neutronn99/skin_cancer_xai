# Explainable AI-Based Early Risk Assessment System for Skin Cancer Using Multimodal Health Data

B.Tech CSE VII Sem - NTCC project (Amity School of Engineering & Technology).
Multimodal (dermoscopic image + clinical metadata) classifier on HAM10000 with Grad-CAM and SHAP explanations.

## What it does
- **Image stream:** EfficientNetB0 transfer-learning CNN (baseline, WPR-04).
- **Clinical stream:** XGBoost on age / sex / lesion site (WPR-05).
- **Fusion:** CNN image embedding concatenated with a clinical embedding -> classifier (WPR-06).
- **Risk score:** `P(akiec) + P(bcc) + P(mel)` from the 7-class output.
- **XAI:** Grad-CAM on the last conv layer (images), SHAP (TreeExplainer) on the clinical model (WPR-07).
- **Evaluation:** lesion-grouped stratified 5-fold CV, ablation (image vs clinical vs fusion), ROC / PR / confusion matrices, untouched hold-out test set (WPR-08 to 10).
- **Demo:** Streamlit app (upload image, enter age/sex/site -> risk, heatmap, SHAP bars).

## Layout
```
src/config.py          hyperparameters, paths, class names
src/data.py            metadata, ClinicalPreprocessor, lesion-grouped splits, tf.data + augmentation
src/models.py          image CNN, fusion network, XGBoost, two-stage training
src/pipeline.py        run_cv / train_final
src/evaluate.py        metrics, CV summary, ablation table, plots
src/xai.py             Grad-CAM, SHAP, gallery
src/run_experiments.py CLI
src/infer.py           Predictor used by the app
app/app.py             Streamlit demo
tests/smoke_test.py    end-to-end test on synthetic data (no dataset needed)
```

## Get the data
Download HAM10000 (Kaggle `kmader/skin-cancer-mnist-ham10000`, or ISIC / Harvard Dataverse) and arrange:
```
data/HAM10000/HAM10000_metadata.csv
data/HAM10000/HAM10000_images_part_1/*.jpg
data/HAM10000/HAM10000_images_part_2/*.jpg
```
(images are found recursively, so the exact sub-folder names don't matter)

## Run
```bash
pip install -r requirements.txt
python tests/smoke_test.py                 # sanity check, ~3 min on CPU
python -m src.run_experiments --stage cv   # CV + ablation  -> results/
python -m src.run_experiments --stage final  # hold-out test, saves models/, Grad-CAM gallery, SHAP plot
streamlit run app/app.py
```
Use a GPU (Google Colab T4): CV for three modalities x 5 folds takes a while. To iterate quickly, lower `epochs_*` / `n_folds` in `src/config.py` or run `--modalities fusion`.

## Methodology notes (put these in your report)
- **No leakage:** splits are grouped by `lesion_id` (HAM10000 has several images per lesion); augmentation and clinical preprocessing (imputation, scaling, one-hot) are fit on training data only; early stopping uses an inner validation split, never the evaluation fold.
- **Imbalance:** balanced class weights + augmentation; report macro-F1, balanced accuracy and melanoma recall, not accuracy alone.
- **Missing age:** median imputation + `age_missing` flag.
- **Hold-out test:** ~1/7 of lesions, touched only in the `final` stage.

## Limitations (state them honestly)
- SHAP explains the **clinical XGBoost model**, not the fusion network's clinical branch; Grad-CAM explains the fusion/image CNN. The app labels this.
- Risk score is uncalibrated; the Low/Moderate/High bands in the app are illustrative.
- HAM10000 is single-source with limited skin-tone diversity; results will not generalise automatically to other populations or to real clinics.
- Not a medical device. Do not use for diagnosis.
- The Grad-CAM vs dermatologist-annotation comparison mentioned in your WPRs is **not** implemented: HAM10000 ships no such annotations. Either obtain lesion masks (e.g. the ISIC segmentation set) and add an IoU check, or reword that WPR line.

## About your WPR numbers
This code produces the metrics; none of the figures in the WPRs come from it yet. Run the experiments and report what you actually get. Fill the results chapter from `results/cv_ablation.md`, `results/test_metrics.csv` and the PNGs.
