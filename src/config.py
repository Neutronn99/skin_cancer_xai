"""Central configuration. Edit values here (or override `cfg` fields in code)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Cfg:
    # paths
    data_dir: Path = ROOT / "data" / "HAM10000"
    results_dir: Path = ROOT / "results"
    models_dir: Path = ROOT / "models"
    # image / training
    img_size: int = 224
    batch_size: int = 32
    epochs_head: int = 5          # backbone frozen
    epochs_finetune: int = 10     # backbone unfrozen (BatchNorm kept frozen)
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    weights: Optional[str] = "imagenet"   # None = random init (used by smoke test)
    # evaluation protocol
    n_folds: int = 5              # CV folds on the development set
    holdout_splits: int = 7       # 1/7 of lesions (~14%) held out as final test set
    inner_splits: int = 5         # 1/5 of training fold used for early stopping
    seed: int = 42
    # labels
    classes: Tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    # classes counted as "malignant / pre-malignant" for the early-risk score
    malignant: Tuple[str, ...] = ("akiec", "bcc", "mel")


cfg = Cfg()
