"""Sjekkar om etikettane i datasettet stemmer med «raud = moden».

Datasettet merkte modning ut frå både farge og fotodato. Viss mange raude eple er
merkte «umoden» (t.d. fordi dei vart fotograferte tidleg i sesongen), lærer
modellen at raudt berre delvis tyder moden, og blir usikker på raude eple.

    python src/check_labels.py

Skriv ut fordelinga av raudandel per klasse, og lagar reports/label_check.png med
dei raudaste «umodne» og dei minst raude «modne» epla, så de kan sjå sjølve.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATA, ROOT, labels_and_groups, load_features, load_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.data)
    y, _ = labels_and_groups(meta)
    red = load_features(meta, "hue", args.data)["red_fraction"].to_numpy()

    print("Raudandel per klasse (persentilar):")
    print(f"  {'':8s} {'5 %':>6s} {'25 %':>6s} {'median':>7s} {'75 %':>6s} {'95 %':>6s}")
    for name, v in [("moden", 1), ("umoden", 0)]:
        q = np.percentile(red[y == v], [5, 25, 50, 75, 95])
        print(f"  {name:8s} " + " ".join(f"{x:6.0%}" for x in q))

    print("\nAndel eple som er merkte «moden», etter kor raude dei er:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (red >= lo) & (red < hi)
        if m.sum():
            print(f"  raudandel {lo:4.0%}–{min(hi, 1):4.0%}: {y[m].mean():5.0%} modne  ({m.sum():5d} eple)")

    # Raudaste «umodne» og minst raude «modne»
    unripe_red = np.where(y == 0)[0][np.argsort(-red[y == 0])][:8]
    ripe_green = np.where(y == 1)[0][np.argsort(red[y == 1])][:8]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for row, (idx, title) in enumerate([(unripe_red, "merkt umoden"), (ripe_green, "merkt moden")]):
        for ax, i in zip(axes[row], idx):
            ax.imshow(Image.open(args.data / meta["file"].iloc[i]))
            ax.set_title(f"{title}\nraud {red[i]:.0%}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Øvst: raudaste eple merkte umodne. Nedst: minst raude eple merkte modne.")
    fig.tight_layout()
    fig.savefig(args.reports / "label_check.png", dpi=130)
    print(f"\nLagra {args.reports / 'label_check.png'}")


if __name__ == "__main__":
    main()
