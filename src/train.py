"""Train and compare fraud-risk models, with several ways of handling the class imbalance.

Split (by time):  train 2019-01..2020-02  |  validation 2020-03..2020-06  |  test 2020-06..2020-12
Model selection and every threshold choice use VALIDATION only; the test set is touched once at the end.

Run:  python -m src.train     -> models/*.joblib, reports/model_comparison.csv, data/scores.pkl
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURES, ROOT, load

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SCORES = ROOT / "data" / "scores.pkl"
NUMERIC = [f for f in FEATURES if f != "category"]
# trees use the category as a number: its fraud rate in the TRAINING period (target encoding, no leakage)
MODEL_FEATURES = [("category_risk" if f == "category" else f) for f in FEATURES]
SEED = 42


def logistic(class_weight=None):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
                             ("num", StandardScaler(), NUMERIC)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight=class_weight))


def boosting(class_weight=None):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=50, l2_regularization=1.0,
        class_weight=class_weight, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)


def undersample(X, y, ratio=10, seed=SEED):
    """Keep every fraud and a random `ratio` legit transactions per fraud."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = rng.choice(np.flatnonzero(y == 0), size=min(len(pos) * ratio, (y == 0).sum()), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return X.iloc[idx], y[idx]


def candidates():
    """name -> (estimator factory, imbalance strategy, feature list)"""
    return {
        "Logistic regression · balanced weights": (lambda: logistic("balanced"), "weights", FEATURES),
        "Gradient boosting · no imbalance handling": (lambda: boosting(None), "none", MODEL_FEATURES),
        "Gradient boosting · balanced weights": (lambda: boosting("balanced"), "weights", MODEL_FEATURES),
        "Gradient boosting · undersampling 1:10": (lambda: boosting(None), "undersample", MODEL_FEATURES),
    }


def run():
    df = load()
    cat_risk = df[df["split"] == "train"].groupby("category", observed=True)["is_fraud"].mean()
    df["category_risk"] = df["category"].map(cat_risk).astype(float)
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    ALL = FEATURES + ["category_risk"]
    Xtr, ytr = tr[ALL], tr["is_fraud"].to_numpy()
    Xva, yva = va[ALL], va["is_fraud"].to_numpy()
    Xte, yte = te[ALL], te["is_fraud"].to_numpy()
    print(f"train {len(tr):,} ({ytr.mean():.2%} fraud) | val {len(va):,} | test {len(te):,} ({yte.mean():.2%} fraud)")

    rows, fitted = [], {}
    # two non-ML baselines
    rows.append({"Model": "Always 'legit' (no model)", "Imbalance": "—", "val_PR_AUC": yva.mean(),
                 "val_ROC_AUC": 0.5, "train_seconds": 0})
    rows.append({"Model": "Rule: rank by amount", "Imbalance": "—",
                 "val_PR_AUC": average_precision_score(yva, Xva["amt"]),
                 "val_ROC_AUC": roc_auc_score(yva, Xva["amt"]), "train_seconds": 0})

    cols_of = {}
    for name, (make, strategy, cols) in candidates().items():
        X, y = (undersample(Xtr[cols], ytr) if strategy == "undersample" else (Xtr[cols], ytr))
        t0 = time.time()
        model = make().fit(X, y)
        cols_of[name] = cols
        p = model.predict_proba(Xva[cols])[:, 1]
        rows.append({"Model": name, "Imbalance": strategy, "val_PR_AUC": average_precision_score(yva, p),
                     "val_ROC_AUC": roc_auc_score(yva, p), "train_seconds": round(time.time() - t0, 1)})
        fitted[name] = model
        print(f"  {name:45s} val PR-AUC {rows[-1]['val_PR_AUC']:.4f}  ({rows[-1]['train_seconds']}s)")

    comp = pd.DataFrame(rows)
    best = comp.iloc[2:].sort_values("val_PR_AUC", ascending=False).iloc[0]["Model"]
    model = fitted[best]
    print(f"\nSelected on validation PR-AUC: {best}")

    # test-set numbers for every model (reported, never used for choosing)
    comp["test_PR_AUC"] = [yte.mean(), average_precision_score(yte, Xte["amt"])] + [
        average_precision_score(yte, fitted[m].predict_proba(Xte[cols_of[m]])[:, 1]) for m in comp["Model"].iloc[2:]]
    comp["test_ROC_AUC"] = [0.5, roc_auc_score(yte, Xte["amt"])] + [
        roc_auc_score(yte, fitted[m].predict_proba(Xte[cols_of[m]])[:, 1]) for m in comp["Model"].iloc[2:]]

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "fraud_model.joblib")
    (MODELS / "model_info.json").write_text(json.dumps(
        {"model": best, "features": cols_of[best], "category_risk": cat_risk.round(5).to_dict()}, indent=2))
    comp.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)

    keep = ["trans_num", "time", "cc_num", "merchant", "city", "state", "is_fraud", "split"] + ALL
    cols = cols_of[best]
    scores = pd.concat([va[keep].assign(score=model.predict_proba(Xva[cols])[:, 1]),
                        te[keep].assign(score=model.predict_proba(Xte[cols])[:, 1])])
    with open(SCORES, "wb") as f:
        pickle.dump(scores, f)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
