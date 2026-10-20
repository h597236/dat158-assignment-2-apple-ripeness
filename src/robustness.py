"""Kor godt toler trekksetta at bileta kjem frå eit anna kamera eller lys?

Modellen er trena på Fuji-eple frå éin frukthage, men nettsida får bilete frå
kvar som helst. Dette skriptet trenar éin random forest per trekksett på
treningsdelen, og testar på testdelen etter at bileta er endra slik eit anna
kamera eller lys ville gjort: lysare, mørkare, sterkare fargar, varmt og kaldt lys.

    python src/robustness.py

Resultat: reports/robustness.csv og reports/robustness.png
Eit trekksett som held F1 oppe under endringane, generaliserer betre til bilete
frå nettsida, sjølv om det ikkje vinn den vanlege kryssvalideringa i train.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATA, RANDOM_STATE, ROOT, group_split, labels_and_groups, load_features, load_meta  # noqa: E402
from features import extract  # noqa: E402


def color_cast(factors):
    def f(img):
        a = np.asarray(img, dtype=np.float32) * np.array(factors)
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    return f


SHIFTS = {
    "original": lambda im: im,
    "lysare (x1.5)": lambda im: ImageEnhance.Brightness(im).enhance(1.5),
    "mørkare (x0.6)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "sterkare fargar (x1.6)": lambda im: ImageEnhance.Color(im).enhance(1.6),
    "blassare fargar (x0.6)": lambda im: ImageEnhance.Color(im).enhance(0.6),
    "varmt lys": color_cast([1.15, 1.0, 0.8]),
    "kaldt lys": color_cast([0.85, 1.0, 1.2]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--feature-sets", default="color,hue,simple,pixels")
    ap.add_argument("--max-test", type=int, default=2000,
                    help="Maks tal testeple (endringane må reknast ut på nytt for kvart bilete)")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    feature_sets = args.feature_sets.split(",")

    meta = load_meta(args.data)
    y, groups = labels_and_groups(meta)
    train_idx, test_idx = group_split(y, groups)
    if len(test_idx) > args.max_test:
        test_idx = np.random.default_rng(RANDOM_STATE).choice(test_idx, args.max_test, replace=False)
    test_imgs = [Image.open(args.data / meta["file"].iloc[i]).convert("RGB") for i in test_idx]

    rows = []
    for fs in feature_sets:
        X = load_features(meta, fs, args.data)  # same trekk og cache som train.py
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)
        model.fit(X.iloc[train_idx], y[train_idx])
        for name, shift in SHIFTS.items():
            X_test = pd.DataFrame(np.stack([extract(shift(im), fs) for im in test_imgs]), columns=X.columns)
            f1 = f1_score(y[test_idx], model.predict(X_test))
            rows.append({"trekk": fs, "endring": name, "f1": f1})
            print(f"  {fs:7s} {name:24s} F1 = {f1:.3f}")

    df = pd.DataFrame(rows)
    table = df.pivot(index="endring", columns="trekk", values="f1").loc[list(SHIFTS)]
    table.loc["snitt utan original"] = table.drop(index="original").mean()
    table.loc["verste fall"] = table.drop(index=["original", "snitt utan original"]).min()
    print("\n", table.round(3))
    table.round(4).to_csv(args.reports / "robustness.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.loc[list(SHIFTS)].plot.bar(ax=ax)
    ax.set_ylabel("F1 for «moden» (testsett)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Robustheit: trekksett under endra lys og kamera")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(args.reports / "robustness.png", dpi=150)
    print(f"\nLagra {args.reports / 'robustness.csv'} og robustness.png")


if __name__ == "__main__":
    main()
