"""Choose the alert threshold with a business cost model (on VALIDATION), then report on TEST.

The model never blocks a card by itself: transactions above the threshold go to a human
review queue. The threshold therefore trades analyst workload against missed fraud.

Cost model (editable in the app):
    every alert reviewed by an analyst       -> REVIEW_COST   (analyst time, ~10 minutes)
    every legit customer wrongly alerted     -> + FRICTION_COST (call / temporary hold annoys the customer)
    every fraud NOT alerted (missed)         -> its transaction amount (charged back to the bank)

Run:  python -m src.threshold   -> reports/threshold_comparison.csv, reports/cost_curve.csv, models/threshold.json
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from .train import MODELS, REPORTS, SCORES

REVIEW_COST = 15.0     # $ per alert reviewed
FRICTION_COST = 10.0   # extra $ per false alarm (customer friction)
CAPACITY_PER_DAY = 120  # alerts the review team can handle per day (for the capacity-based option)


def confusion_at(y, s, amt, t, review=REVIEW_COST, friction=FRICTION_COST):
    alert = s >= t
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum()); tn = int((~alert & (y == 0)).sum())
    caught = float(amt[alert & (y == 1)].sum()); missed = float(amt[~alert & (y == 1)].sum())
    cost = review * (tp + fp) + friction * fp + missed
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": t, "alerts": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fraud_$_caught": caught, "fraud_$_missed": missed, "total_cost": cost}


def curve(y, s, amt, grid, **kw):
    return pd.DataFrame([confusion_at(y, s, amt, t, **kw) for t in grid])


def load_scores():
    with open(SCORES, "rb") as f:
        sc = pickle.load(f)
    return sc[sc["split"] == "val"], sc[sc["split"] == "test"]


def run():
    va, te = load_scores()
    days_va = (va["time"].max() - va["time"].min()).days + 1
    days_te = (te["time"].max() - te["time"].min()).days + 1
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(va["score"], 1 - np.logspace(-4, -1.3, 60))]))
    cv = curve(va["is_fraud"].to_numpy(), va["score"].to_numpy(), va["amt"].to_numpy(), grid)
    cv["alerts_per_day"] = cv["alerts"] / days_va

    # candidate thresholds, all chosen on VALIDATION
    choices = {
        "Default 0.5": 0.5,
        "Best F1": float(cv.loc[cv["f1"].idxmax(), "threshold"]),
        "Lowest total cost ⭐": float(cv.loc[cv["total_cost"].idxmin(), "threshold"]),
        f"Fill team capacity ({CAPACITY_PER_DAY}/day)": float(cv.loc[cv["alerts_per_day"] <= CAPACITY_PER_DAY, "threshold"].min()),
    }
    y, s, a = te["is_fraud"].to_numpy(), te["score"].to_numpy(), te["amt"].to_numpy()
    no_model = {"threshold": None, "alerts": 0, "TP": 0, "FP": 0, "FN": int(y.sum()), "TN": int((y == 0).sum()),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "fraud_$_caught": 0.0,
                "fraud_$_missed": float(a[y == 1].sum()), "total_cost": float(a[y == 1].sum())}
    rows = [{"Policy": "No model (review nothing)", **no_model}]
    rows += [{"Policy": k, **confusion_at(y, s, a, t)} for k, t in choices.items()]
    comp = pd.DataFrame(rows)
    comp["alerts_per_day"] = comp["alerts"] / days_te
    comp["savings_vs_no_model"] = comp.loc[0, "total_cost"] - comp["total_cost"]

    REPORTS.mkdir(exist_ok=True)
    comp.round(4).to_csv(REPORTS / "threshold_comparison.csv", index=False)
    test_curve = curve(y, s, a, grid)
    test_curve["alerts_per_day"] = test_curve["alerts"] / days_te
    pd.concat([cv.assign(split="val"), test_curve.assign(split="test")]).round(5).to_csv(
        REPORTS / "cost_curve.csv", index=False)
    rec = {"recommended_threshold": choices["Lowest total cost ⭐"], "review_cost": REVIEW_COST,
           "friction_cost": FRICTION_COST, "chosen_on": "validation (Mar–Jun 2020)", "all_choices": choices}
    (MODELS / "threshold.json").write_text(json.dumps(rec, indent=2))

    pd.set_option("display.width", 200)
    show = ["Policy", "threshold", "alerts_per_day", "precision", "recall", "fraud_$_caught", "fraud_$_missed",
            "total_cost", "savings_vs_no_model"]
    print(f"Test period: {days_te} days, {int(y.sum()):,} frauds worth ${a[y == 1].sum():,.0f}\n")
    print(comp[show].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
