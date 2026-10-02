"""Inference helper shared by the Streamlit app and scripts."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import xgboost as xgb
from PIL import Image

from .config import cfg
from .evaluate import risk_score
from .xai import gradcam, overlay, shap_risk_values


class Predictor:
    def __init__(self):
        self.pre = joblib.load(cfg.models_dir / "preprocessor.joblib")
        self.fusion = tf.keras.models.load_model(cfg.models_dir / "fusion.keras")
        self.bst = xgb.Booster()
        self.bst.load_model(str(cfg.models_dir / "clinical_xgb.json"))
        self.explainer = shap.TreeExplainer(self.bst)

    @property
    def sites(self):
        return [l for l in self.pre.locs]

    def predict(self, pil_image: Image.Image, age, sex: str, site: str) -> dict:
        row = pd.DataFrame({"age": [age], "sex": [sex], "localization": [site]})
        X = self.pre.transform(row)
        img = np.asarray(pil_image.convert("RGB").resize((cfg.img_size, cfg.img_size)), dtype="float32")
        proba = self.fusion.predict({"image": img[None], "clinical": X}, verbose=0)
        cam, top = gradcam(self.fusion, img, X[0])
        sv = shap_risk_values(self.explainer, X)[0]
        return {
            "proba": dict(zip(cfg.classes, proba[0].tolist())),
            "risk": float(risk_score(proba)[0]),
            "top_class": cfg.classes[top],
            "overlay": overlay(img, cam),
            "shap_values": sv,
            "feature_names": self.pre.feature_names,
        }
