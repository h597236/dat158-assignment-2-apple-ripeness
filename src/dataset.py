"""Lasting av utklipp, trekk og trenings-/testsplitt. Brukt av train.py og notatboka,
så begge ser nøyaktig same data og same splitt."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from features import extract, feature_names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "crops"
RANDOM_STATE = 42


def load_meta(data_dir: Path = DATA) -> pd.DataFrame:
    path = data_dir / "metadata.csv"
    if not path.exists():
        raise SystemExit(f"Fann ikkje {path}. Køyr scripts/download_data.py og "
                         "src/prepare_crops.py først (sjå README).")
    return pd.read_csv(path, encoding="utf-8")


def load_features(meta: pd.DataFrame, feature_set: str, data_dir: Path = DATA) -> pd.DataFrame:
    """Reknar ut trekk for alle eple, med cache i data/crops/features_<sett>.npy."""
    cache = data_dir / f"features_{feature_set}.npy"
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(meta):
            return pd.DataFrame(X, columns=feature_names(feature_set))
    print(f"Reknar ut trekk «{feature_set}» for {len(meta)} eple ...")
    X = np.stack([extract(Image.open(data_dir / f), feature_set) for f in meta["file"]])
    np.save(cache, X)
    return pd.DataFrame(X, columns=feature_names(feature_set))


def labels_and_groups(meta: pd.DataFrame):
    y = (meta["label"] == "ripe").astype(int).to_numpy()   # 1 = moden, 0 = umoden
    groups = meta["group"].to_numpy()
    return y, groups


def group_split(y, groups, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Trenings-/testsplitt per foto, så eple frå same foto aldri er på begge sider.

    StratifiedGroupKFold held kvart foto samla og prøver samtidig å gi trening og
    test same andel modne eple. Vi tek éin av foldane som testsett (5 foldar = 20 %).
    """
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    assert not set(groups[train_idx]) & set(groups[test_idx]), "Same foto i trening og test!"
    return train_idx, test_idx
