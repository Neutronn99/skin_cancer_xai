"""Metrics, CV summaries and plots (confusion matrix, ROC / PR for the malignant-risk score)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import cfg

K = len(cfg.classes)
MEL = cfg.classes.index("mel")
MAL = [cfg.classes.index(c) for c in cfg.malignant]


def risk_score(proba: np.ndarray) -> np.ndarray:
    """Early-risk score = P(akiec) + P(bcc) + P(mel)."""
    return proba[:, MAL].sum(1)


def compute_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    proba = proba / np.clip(proba.sum(1, keepdims=True), 1e-8, None)
    pred = proba.argmax(1)
    yb = np.isin(y, MAL).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(K), zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", labels=range(K), zero_division=0),
        "melanoma_recall": recall_score(y == MEL, pred == MEL, zero_division=0),
    }
    for name, fn in [
        ("auc_macro_ovr", lambda: roc_auc_score(y, proba, multi_class="ovr", labels=list(range(K)))),
        ("risk_auc", lambda: roc_auc_score(yb, risk_score(proba))),
        ("risk_avg_precision", lambda: average_precision_score(yb, risk_score(proba))),
    ]:
        try:
            out[name] = fn()
        except ValueError:
            out[name] = float("nan")
    return out


def cv_summary(y: np.ndarray, oof: np.ndarray, folds) -> dict:
    """Mean / std of each metric over the CV folds."""
    rows = pd.DataFrame([compute_metrics(y[va], oof[va]) for _, va in folds])
    return {"mean": rows.mean().to_dict(), "std": rows.std(ddof=0).to_dict(), "per_fold": rows}


def ablation_table(summaries: dict) -> pd.DataFrame:
    """summaries: {modality: cv_summary(...)} -> 'mean ± std' table."""
    data = {}
    for name, s in summaries.items():
        data[name] = {m: f"{s['mean'][m]:.3f} ± {s['std'][m]:.3f}" for m in s["mean"]}
    return pd.DataFrame(data).T


def plot_confusion(y, proba, path, title=""):
    cm = confusion_matrix(y, proba.argmax(1), labels=range(K))
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), cfg.classes, rotation=45)
    ax.set_yticks(range(K), cfg.classes)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title or "Confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_risk_curves(y, probas: dict, path_prefix):
    """ROC and Precision-Recall curves of the malignant-risk score, one line per modality."""
    yb = np.isin(y, MAL).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yb, risk_score(p))
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yb, risk_score(p)):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC - malignant risk score"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{path_prefix}_roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        pr, rc, _ = precision_recall_curve(yb, risk_score(p))
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(yb, risk_score(p)):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - malignant risk score"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(f"{path_prefix}_pr.png", dpi=150); plt.close(fig)
