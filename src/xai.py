"""Explainability: Grad-CAM for the image stream, SHAP for the clinical stream."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from PIL import Image

from .config import cfg
from .models import LAST_CONV


# --------------------------------------------------------------------------- Grad-CAM
def gradcam(model, image: np.ndarray, clin: np.ndarray | None = None, class_idx: int | None = None):
    """Return (heatmap in [0,1] at image resolution, class_idx).

    image: (H, W, 3) float 0..255 ; clin: (n_clin,) for the fusion model.
    Target = class_idx, or the predicted class when None.
    """
    gm = tf.keras.Model(model.inputs, [model.get_layer(LAST_CONV).output, model.output])
    inputs = {"image": tf.convert_to_tensor(image[None], tf.float32)}
    if clin is not None:
        inputs["clinical"] = tf.convert_to_tensor(clin[None], tf.float32)
    with tf.GradientTape() as tape:
        conv, preds = gm(inputs)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2]).numpy()[..., 0]
    return np.clip(cam, 0, 1), class_idx


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3] * 255
    return np.clip((1 - alpha) * image + alpha * colored, 0, 255).astype("uint8")


# --------------------------------------------------------------------------- SHAP
def shap_values_for_class(explainer, X: np.ndarray, class_idx: int) -> np.ndarray:
    """Handles both SHAP output layouts (list of arrays vs. (n, features, classes))."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[class_idx])
    sv = np.asarray(sv)
    return sv[:, :, class_idx] if sv.ndim == 3 else sv


def shap_risk_values(explainer, X: np.ndarray) -> np.ndarray:
    """Sum of SHAP values over the malignant classes (log-odds space, approximate 'risk' attribution)."""
    idx = [cfg.classes.index(c) for c in cfg.malignant]
    return sum(shap_values_for_class(explainer, X, i) for i in idx)


def plot_shap_summary(explainer, X, feature_names, path, class_name="mel"):
    sv = shap_values_for_class(explainer, X, cfg.classes.index(class_name))
    plt.figure()
    shap.summary_plot(sv, X, feature_names=feature_names, show=False)
    plt.title(f"SHAP summary - clinical model, class '{class_name}'")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def shap_bar_figure(sv_row: np.ndarray, feature_names, top_k: int = 8):
    order = np.argsort(np.abs(sv_row))[::-1][:top_k][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh([feature_names[i] for i in order], sv_row[order],
            color=["tab:red" if v > 0 else "tab:blue" for v in sv_row[order]])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP value (pushes risk up / down)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- gallery
def make_gallery(model, df, X_clin, out_path, n: int = 8, fusion: bool = True):
    """Grad-CAM panel for n test lesions (one per class where possible)."""
    rng = np.random.default_rng(cfg.seed)
    picks = []
    for c in range(len(cfg.classes)):
        idx = np.where(df["y"].values == c)[0]
        if len(idx):
            picks.append(int(rng.choice(idx)))
    picks = picks[:n]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.6), squeeze=False)
    for col, i in enumerate(picks):
        img = np.asarray(Image.open(df["path"].iloc[i]).convert("RGB")
                         .resize((cfg.img_size, cfg.img_size)), dtype="float32")
        cam, pred = gradcam(model, img, X_clin[i] if fusion else None)
        axes[0, col].imshow(img.astype("uint8")); axes[1, col].imshow(overlay(img, cam))
        axes[0, col].set_title(f"true: {cfg.classes[df['y'].iloc[i]]}", fontsize=8)
        axes[1, col].set_title(f"pred: {cfg.classes[pred]}", fontsize=8)
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
