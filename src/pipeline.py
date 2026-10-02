"""Cross-validation and final training for the three modalities (image / clinical / fusion)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import cfg
from .data import (ClinicalPreprocessor, class_weights, group_folds, inner_split, make_ds)
from .models import build_fusion_model, build_image_model, fit_model, train_xgb

K = len(cfg.classes)


def fit_predict(modality: str, dtr: pd.DataFrame, Xtr: np.ndarray, dte: pd.DataFrame, Xte: np.ndarray):
    """Train on (dtr, Xtr) and return (probabilities on dte, trained model)."""
    if modality == "clinical":
        model = train_xgb(Xtr, dtr["y"].values)
        return model.predict_proba(Xte), model

    tf.keras.backend.clear_session()
    tr_i, va_i = inner_split(dtr)                     # inner validation for early stopping
    a, b = dtr.iloc[tr_i], dtr.iloc[va_i]
    model = build_image_model(K) if modality == "image" else build_fusion_model(Xtr.shape[1], K)
    fit_model(model, make_ds(a, Xtr[tr_i], modality, True),
              make_ds(b, Xtr[va_i], modality, False), class_weights(a["y"].values))
    return model.predict(make_ds(dte, Xte, modality, False), verbose=0), model


def run_cv(dev: pd.DataFrame, modality: str):
    """Lesion-grouped, stratified K-fold. Returns (out-of-fold probabilities, folds)."""
    folds = group_folds(dev, cfg.n_folds)
    oof = np.zeros((len(dev), K), dtype="float32")
    for k, (tr, va) in enumerate(folds):
        print(f"[{modality}] fold {k + 1}/{len(folds)}")
        dtr, dva = dev.iloc[tr].reset_index(drop=True), dev.iloc[va].reset_index(drop=True)
        pre = ClinicalPreprocessor().fit(dtr)          # fit on the training fold only
        proba, _ = fit_predict(modality, dtr, pre.transform(dtr), dva, pre.transform(dva))
        oof[va] = proba
    return oof, folds


def train_final(dev: pd.DataFrame, test: pd.DataFrame, modality: str):
    """Train on the whole development set, predict the untouched hold-out test set."""
    pre = ClinicalPreprocessor().fit(dev)
    Xdev, Xtest = pre.transform(dev), pre.transform(test)
    proba, model = fit_predict(modality, dev, Xdev, test, Xtest)
    return proba, model, pre, Xtest
