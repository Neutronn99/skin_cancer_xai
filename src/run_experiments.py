"""Command line entry point.

    python -m src.run_experiments --stage cv      # 5-fold CV + ablation table
    python -m src.run_experiments --stage final   # train on dev set, evaluate hold-out, save models, XAI
    python -m src.run_experiments --stage all
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import cfg
from .data import load_metadata, split_holdout
from .evaluate import (ablation_table, compute_metrics, cv_summary, plot_confusion, plot_risk_curves)
from .pipeline import run_cv, train_final
from .xai import make_gallery, plot_shap_summary
import shap

MODS = ["image", "clinical", "fusion"]


def cv_stage(dev, mods):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    y = dev["y"].values
    summaries, oofs = {}, {}
    for m in mods:
        oof, folds = run_cv(dev, m)
        np.save(cfg.results_dir / f"oof_{m}.npy", oof)
        summaries[m] = cv_summary(y, oof, folds)
        summaries[m]["per_fold"].to_csv(cfg.results_dir / f"cv_folds_{m}.csv", index=False)
        oofs[m] = oof
        plot_confusion(y, oof, cfg.results_dir / f"cv_confusion_{m}.png", f"CV confusion - {m}")
    table = ablation_table(summaries)
    table.to_csv(cfg.results_dir / "cv_ablation.csv")
    (cfg.results_dir / "cv_ablation.md").write_text(table.to_markdown() if _has_tabulate() else table.to_string())
    plot_risk_curves(y, oofs, str(cfg.results_dir / "cv"))
    print(f"\n=== {cfg.n_folds}-fold CV ablation ===\n", table.to_string())


def final_stage(dev, test, mods):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    yt = test["y"].values
    rows, probas = {}, {}
    for m in mods:
        proba, model, pre, Xtest = train_final(dev, test, m)
        probas[m] = proba
        rows[m] = compute_metrics(yt, proba)
        np.save(cfg.results_dir / f"test_proba_{m}.npy", proba)
        plot_confusion(yt, proba, cfg.results_dir / f"test_confusion_{m}.png", f"Test confusion - {m}")
        if m == "clinical":
            model.get_booster().save_model(str(cfg.models_dir / "clinical_xgb.json"))
            joblib.dump(pre, cfg.models_dir / "preprocessor.joblib")
            explainer = shap.TreeExplainer(model.get_booster())
            plot_shap_summary(explainer, Xtest, pre.feature_names, cfg.results_dir / "shap_summary_mel.png", "mel")
        else:
            model.save(cfg.models_dir / f"{m}.keras")
            if m == "fusion":
                make_gallery(model, test, Xtest, cfg.results_dir / "gradcam_gallery.png", fusion=True)
            else:
                make_gallery(model, test, Xtest, cfg.results_dir / "gradcam_gallery_image.png", fusion=False)
    table = pd.DataFrame(rows).T.round(3)
    table.to_csv(cfg.results_dir / "test_metrics.csv")
    plot_risk_curves(yt, probas, str(cfg.results_dir / "test"))
    json.dump({"n_dev": len(dev), "n_test": len(test)}, open(cfg.results_dir / "split_sizes.json", "w"))
    print("\n=== Hold-out test metrics ===\n", table.to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "final", "all"], default="all")
    ap.add_argument("--modalities", nargs="+", choices=MODS, default=MODS)
    args = ap.parse_args()
    df = load_metadata()
    dev, test = split_holdout(df)
    print(f"images: {len(df)} | dev: {len(dev)} | test: {len(test)}")
    if args.stage in ("cv", "all"):
        cv_stage(dev, args.modalities)
    if args.stage in ("final", "all"):
        final_stage(dev, test, args.modalities)


if __name__ == "__main__":
    main()
