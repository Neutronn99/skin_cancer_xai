"""Model builders: image CNN, clinical XGBoost, multimodal fusion network."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from .config import cfg

L = tf.keras.layers
LAST_CONV = "top_activation"  # last conv feature map of EfficientNetB0 (used by Grad-CAM)


def _backbone(img_in):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=cfg.weights, input_tensor=img_in
    )
    names = [l.name for l in base.layers if not isinstance(l, L.InputLayer)]
    return base, names


def _image_branch(img_in):
    base, names = _backbone(img_in)
    x = L.GlobalAveragePooling2D()(base.output)
    x = L.Dropout(0.3)(x)
    emb = L.Dense(128, activation="relu", name="img_embedding")(x)
    return emb, names


def build_image_model(n_classes: int) -> tf.keras.Model:
    """Image-only CNN baseline (EfficientNetB0 transfer learning)."""
    img_in = L.Input((cfg.img_size, cfg.img_size, 3), name="image")
    emb, names = _image_branch(img_in)
    out = L.Dense(n_classes, activation="softmax", name="probs")(L.Dropout(0.3)(emb))
    model = tf.keras.Model(img_in, out, name="image_cnn")
    model.backbone_names = tuple(names)
    return model


def build_fusion_model(n_clin: int, n_classes: int) -> tf.keras.Model:
    """Early-fusion: CNN image embedding || clinical embedding -> concatenation -> classifier."""
    img_in = L.Input((cfg.img_size, cfg.img_size, 3), name="image")
    clin_in = L.Input((n_clin,), name="clinical")
    img_emb, names = _image_branch(img_in)
    c = L.Dense(32, activation="relu")(clin_in)
    c = L.Dense(16, activation="relu", name="clin_embedding")(c)
    x = L.Concatenate(name="fusion_concat")([img_emb, c])
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model([img_in, clin_in], out, name="multimodal_fusion")
    model.backbone_names = tuple(names)
    return model


# --------------------------------------------------------------------------- training
def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _set_backbone_trainable(model, flag: bool):
    for n in model.backbone_names:
        layer = model.get_layer(n)
        # BatchNorm stays frozen during fine-tuning (standard practice with small batches)
        layer.trainable = flag and not isinstance(layer, L.BatchNormalization)


def fit_model(model, train_ds, val_ds, class_weight):
    """Two-stage training: frozen backbone (head) -> fine-tune backbone at low LR."""
    _set_backbone_trainable(model, False)
    _compile(model, cfg.lr_head)
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_head,
              class_weight=class_weight, verbose=2)
    if cfg.epochs_finetune > 0:
        _set_backbone_trainable(model, True)
        _compile(model, cfg.lr_finetune)
        es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                              monitor="val_loss")
        model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_finetune,
                  class_weight=class_weight, callbacks=[es], verbose=2)
    return model


def train_xgb(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """Clinical-only gradient-boosted trees with balanced sample weights."""
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="mlogloss", random_state=cfg.seed, n_jobs=-1,
    )
    clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return clf
