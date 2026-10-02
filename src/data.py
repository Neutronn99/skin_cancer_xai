"""Data loading, clinical preprocessing, lesion-level splitting, tf.data pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import cfg

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- metadata
def load_metadata() -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach image paths + integer labels."""
    csv = cfg.data_dir / "HAM10000_metadata.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. See README 'Get the data'.")
    df = pd.read_csv(csv)
    paths = {p.stem: str(p) for p in cfg.data_dir.rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["y"] = df["dx"].map({c: i for i, c in enumerate(cfg.classes)}).astype(int)
    return df


class ClinicalPreprocessor:
    """Age (impute + scale + missing flag), sex and lesion site (one-hot).

    Fit ONLY on training data of each fold so nothing leaks from validation/test.
    """

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        age = df["age"].astype(float)
        self.age_median = float(age.median())
        filled = age.fillna(self.age_median)
        self.mu = float(filled.mean())
        self.sd = float(filled.std() or 1.0)
        self.sexes = ["male", "female", "unknown"]
        self.locs = sorted(set(df["localization"].fillna("unknown")) | {"unknown"})
        self.feature_names = (
            ["age", "age_missing"]
            + [f"sex_{s}" for s in self.sexes]
            + [f"site_{l}" for l in self.locs]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        age = df["age"].astype(float)
        missing = age.isna().astype(float).values
        age = ((age.fillna(self.age_median) - self.mu) / self.sd).values
        cols = [age, missing]
        sex = df["sex"].fillna("unknown").values
        cols += [(sex == s).astype(float) for s in self.sexes]
        loc = df["localization"].fillna("unknown").values
        cols += [(loc == l).astype(float) for l in self.locs]
        return np.stack(cols, axis=1).astype("float32")


# --------------------------------------------------------------------------- splitting
def group_folds(df: pd.DataFrame, n_splits: int):
    """Stratified folds grouped by lesion_id (several images of one lesion never straddle a split)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    return list(sgkf.split(df, df["y"], groups=df["lesion_id"]))


def split_holdout(df: pd.DataFrame):
    dev_idx, test_idx = group_folds(df, cfg.holdout_splits)[0]
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def inner_split(df: pd.DataFrame):
    return group_folds(df, cfg.inner_splits)[0]


def class_weights(y: np.ndarray) -> dict:
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    d = {i: 1.0 for i in range(len(cfg.classes))}
    d.update({int(c): float(x) for c, x in zip(present, w)})
    return d


# --------------------------------------------------------------------------- tf.data
def _load_img(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (cfg.img_size, cfg.img_size))
    return tf.cast(img, tf.float32)  # 0..255 (EfficientNet rescales internally)


def _augment(img):
    """Train-time augmentation only (applied after the split, never to val/test)."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, 20.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0)


def make_ds(df: pd.DataFrame, X_clin: np.ndarray, modality: str, training: bool):
    """modality: 'image' or 'fusion'. Inputs are dicts keyed by the model's input names."""
    paths = df["path"].values.astype(str)
    y = df["y"].values.astype("int32")

    def build(path, clin, label):
        img = _load_img(path)
        if training:
            img = _augment(img)
        inputs = {"image": img}
        if modality == "fusion":
            inputs["clinical"] = clin
        return inputs, label

    ds = tf.data.Dataset.from_tensor_slices((paths, X_clin, y))
    if training:
        ds = ds.shuffle(len(df), seed=cfg.seed)
    return ds.map(build, num_parallel_calls=AUTOTUNE).batch(cfg.batch_size).prefetch(AUTOTUNE)
