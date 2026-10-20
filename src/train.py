"""Trenar og evaluerer modningsklassifikatoren, og lagrar modellen til nettsida.

    python src/train.py              # full køyring
    python src/train.py --quick      # mindre hyperparametersøk, for rask testing

Resultat:
    models/apple_model.joblib        modellen som app.py brukar
    reports/model_comparison.csv     kryssvalidert F1 for alle modellar og trekksett
    reports/metrics.json             val, terskel og testresultat
    reports/*.png                    figurar til rapporten

Stega følgjer ML-livssyklusen frå DAT158: baseline -> samanlikning med
kryssvalidering -> hyperparametersøk -> val av terskel -> éin endeleg test.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # lagre figurar utan skjerm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             precision_recall_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DATA, RANDOM_STATE, ROOT, group_split, labels_and_groups, load_features, load_meta  # noqa: E402

CLASS_NAMES = ["umoden", "moden"]
FEATURE_SETS = ["color", "pixels"]


def candidate_models():
    return {
        "Logistisk regresjon": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(random_state=RANDOM_STATE)),
        "Random forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "SVC (RBF)": make_pipeline(StandardScaler(), SVC()),
    }


def search_spaces(quick: bool):
    if quick:
        rf = {"n_estimators": [100], "max_depth": [None, 10]}
        svc = {"svc__C": [1], "svc__gamma": ["scale"]}
    else:
        rf = {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]}
        svc = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]}
    return {
        "Random forest": (RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE), rf),
        "SVC (RBF)": (make_pipeline(StandardScaler(), SVC()), svc),
    }


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--model-out", type=Path, default=ROOT / "models" / "apple_model.joblib")
    ap.add_argument("--target-precision", type=float, default=0.95,
                    help="Minste precision for «moden» når terskelen blir vald")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    section("Data")
    meta = load_meta(args.data)
    y, groups = labels_and_groups(meta)
    X_all = {fs: load_features(meta, fs, args.data) for fs in FEATURE_SETS}
    train_idx, test_idx = group_split(y, groups)
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]
    print(f"Trening: {len(train_idx)} eple frå {len(set(g_train))} foto, {y_train.mean():.0%} modne")
    print(f"Test:    {len(test_idx)} eple frå {len(set(groups[test_idx]))} foto, {y_test.mean():.0%} modne")

    # GroupKFold: kvar fold held på heile foto, same grunn som for testsplitten
    cv = GroupKFold(n_splits=args.folds)

    # ------------------------------------------------------------ baseline
    section("Baseline")
    dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), X_all["color"].iloc[train_idx],
                            y_train, groups=g_train, cv=cv, scoring="f1")
    print(f"DummyClassifier F1 = {dummy.mean():.3f} ± {dummy.std():.3f}")

    # --------------------------------------------------------- samanlikning
    section("Samanlikning (kryssvalidert F1 for «moden»)")
    rows = [{"trekk": "-", "modell": "Dummy (mest vanlege klasse)",
             "f1_mean": dummy.mean(), "f1_std": dummy.std()}]
    for fs in FEATURE_SETS:
        X_train = X_all[fs].iloc[train_idx]
        for name, model in candidate_models().items():
            s = cross_val_score(model, X_train, y_train, groups=g_train, cv=cv, scoring="f1", n_jobs=-1)
            rows.append({"trekk": fs, "modell": name, "f1_mean": s.mean(), "f1_std": s.std()})
            print(f"  {fs:7s} {name:22s} {s.mean():.3f} ± {s.std():.3f}")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.reports / "model_comparison.csv", index=False)

    best_fs = comparison[comparison["trekk"] != "-"].groupby("trekk")["f1_mean"].max().idxmax()
    X_train, X_test = X_all[best_fs].iloc[train_idx], X_all[best_fs].iloc[test_idx]
    print(f"\nBeste trekksett: {best_fs}")

    # --------------------------------------------------- hyperparametersøk
    section("Hyperparametersøk (GridSearchCV)")
    grids = {}
    for name, (estimator, params) in search_spaces(args.quick).items():
        grid = GridSearchCV(estimator, params, cv=cv, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train, groups=g_train)
        grids[name] = grid
        print(f"  {name:15s} F1 = {grid.best_score_:.3f}  {grid.best_params_}")
    best_name = max(grids, key=lambda n: grids[n].best_score_)
    print(f"\nVald modell: {best_name}")

    final_model = clone(grids[best_name].best_estimator_)
    if best_name.startswith("SVC"):
        # SVC gir ikkje sannsyn direkte; kalibrering gjer om poenga til sannsyn,
        # som nettsida viser og terskelen blir sett på.
        final_model = CalibratedClassifierCV(final_model, ensemble=False)

    # ------------------------------------------------------ val av terskel
    section("Val av terskel")
    # Kryssvaliderte sannsyn på treningssettet, så testsettet ikkje blir brukt til å velje.
    proba_cv = cross_val_predict(final_model, X_train, y_train, groups=g_train,
                                 cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    print("Kryssvalidert, terskel 0.5:")
    print(classification_report(y_train, (proba_cv >= 0.5).astype(int), target_names=CLASS_NAMES))

    precision, recall, thresholds = precision_recall_curve(y_train, proba_cv)
    ok = np.where(precision[:-1] >= args.target_precision)[0]
    if len(ok):
        threshold = float(thresholds[ok[0]])
        print(f"Terskel {threshold:.3f} gir precision {precision[ok[0]]:.3f}, recall {recall[ok[0]]:.3f}")
    else:
        threshold = 0.5
        print(f"Nådde ikkje precision {args.target_precision}; brukar 0.5")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, precision[:-1], label="precision")
    ax.plot(thresholds, recall[:-1], label="recall")
    ax.axvline(threshold, color="grey", linestyle="--", label=f"vald terskel {threshold:.2f}")
    ax.set_xlabel("Terskel for «moden»")
    ax.set_title("Precision og recall (kryssvalidert på treningssettet)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.reports / "precision_recall_threshold.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- endeleg test
    section("Endeleg test (berre éin gong)")
    final_model.fit(X_train, y_train)
    proba_test = final_model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    report_05 = classification_report(y_test, (proba_test >= 0.5).astype(int),
                                      target_names=CLASS_NAMES, output_dict=True)
    report_t = classification_report(y_test, pred_test, target_names=CLASS_NAMES, output_dict=True)
    print(f"Terskel {threshold:.2f}:")
    print(classification_report(y_test, pred_test, target_names=CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred_test, display_labels=CLASS_NAMES, ax=ax, colorbar=False)
    ax.set_title(f"Testsett, terskel {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(args.reports / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    # Feilklassifiserte eple, til feilanalyse i rapporten
    wrong = np.where(pred_test != y_test)[0][:8]
    if len(wrong):
        fig, axes = plt.subplots(1, len(wrong), figsize=(2.2 * len(wrong), 2.8), squeeze=False)
        for ax, i in zip(axes[0], wrong):
            row = meta.iloc[test_idx[i]]
            ax.imshow(Image.open(args.data / row["file"]))
            ax.set_title(f"fasit: {row['label']}\nP(moden)={proba_test[i]:.2f}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.reports / "misclassified_test.png", dpi=150)
        plt.close(fig)

    if best_name == "Random forest":
        imp = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(ax=ax)
        ax.set_title("Viktigaste trekk (random forest)")
        fig.tight_layout()
        fig.savefig(args.reports / "feature_importance.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------- lagring
    joblib.dump({"model": final_model, "model_name": best_name,
                 "feature_set": best_fs, "threshold": threshold}, args.model_out)

    metrics = {
        "feature_set": best_fs, "model": best_name,
        "best_params": grids[best_name].best_params_,
        "cv_f1": grids[best_name].best_score_,
        "threshold": threshold, "target_precision": args.target_precision,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "test_at_0.5": report_05, "test_at_threshold": report_t,
    }
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    section("Ferdig")
    print(f"Modell:    {args.model_out}")
    print(f"Rapportar: {args.reports}/")


if __name__ == "__main__":
    main()
