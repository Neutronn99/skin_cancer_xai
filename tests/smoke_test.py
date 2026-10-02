"""End-to-end smoke test on SYNTHETIC data (tiny images, 1 epoch, random init).
Verifies the code runs; the numbers it prints are meaningless.   python tests/smoke_test.py
"""
import sys, tempfile, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

from src.config import cfg

tmp = Path(tempfile.mkdtemp())
data = tmp / "HAM10000"; (data / "imgs").mkdir(parents=True)
rng = np.random.default_rng(0)
rows, k = [], 0
sites = ["back", "face", "foot", "trunk", "unknown"]
for c, cls in enumerate(cfg.classes):
    for l in range(14):
        for j in range(rng.integers(1, 3)):
            iid = f"ISIC_{k:07d}"; k += 1
            base = np.array([30 * c + 20, 200 - 20 * c, 80 + 10 * c])
            arr = np.clip(base + rng.normal(0, 25, (80, 80, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr).save(data / "imgs" / f"{iid}.jpg")
            rows.append(dict(lesion_id=f"HAM_{c}_{l}", image_id=iid, dx=cls, dx_type="histo",
                             age=(np.nan if rng.random() < .05 else float(rng.integers(20, 80))),
                             sex=rng.choice(["male", "female"]), localization=rng.choice(sites)))
pd.DataFrame(rows).to_csv(data / "HAM10000_metadata.csv", index=False)

cfg.data_dir, cfg.results_dir, cfg.models_dir = data, tmp / "results", tmp / "models"
cfg.img_size, cfg.batch_size, cfg.weights = 64, 16, None
cfg.epochs_head, cfg.epochs_finetune = 1, 1
cfg.n_folds, cfg.holdout_splits, cfg.inner_splits = 2, 4, 3

from src import run_experiments as R
from src.data import load_metadata, split_holdout

df = load_metadata(); dev, test = split_holdout(df)
assert set(dev.lesion_id).isdisjoint(set(test.lesion_id)), "lesion leakage!"
print("images", len(df), "dev", len(dev), "test", len(test))
R.cv_stage(dev, R.MODS)
R.final_stage(dev, test, R.MODS)

from src.infer import Predictor
p = Predictor()
out = p.predict(Image.open(df.path[0]), 55, "male", "back")
assert 0 <= out["risk"] <= 1 and out["overlay"].shape[-1] == 3
print("risk", round(out["risk"], 3), "top", out["top_class"])
print("files:", sorted(x.name for x in cfg.results_dir.iterdir()))
print("SMOKE TEST PASSED")
