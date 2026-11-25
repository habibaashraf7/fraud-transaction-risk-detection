"""Simple Python API behind the review console (and usable from any script).

    from src.service import RiskService
    svc = RiskService()
    svc.queue(date="2020-12-24", threshold=0.76)      # alerts for one day, riskiest first, with reasons
    svc.policy_metrics(threshold=0.76)                 # precision / recall / $ cost on the test period
    svc.what_if(amt=950, category="shopping_net", hour=2, amt_vs_cust_avg=8)

The model only produces a RISK SCORE and reasons. It never blocks, declines or labels a
transaction as fraud: every alert goes to a human analyst who makes the decision.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from .explain import CATEGORY_NAMES, FEATURE_LABELS, Explainer
from .threshold import FRICTION_COST, REVIEW_COST, confusion_at
from .train import MODELS, REPORTS, SCORES


class RiskService:
    def __init__(self):
        with open(SCORES, "rb") as f:
            sc = pickle.load(f)
        self.test = sc[sc["split"] == "test"].reset_index(drop=True)
        self.val = sc[sc["split"] == "val"].reset_index(drop=True)
        self.days = (self.test["time"].max() - self.test["time"].min()).days + 1
        self.ex = Explainer()
        self.rec = json.loads((MODELS / "threshold.json").read_text())
        self.recommended = float(self.rec["recommended_threshold"])
        self._y = self.test["is_fraud"].to_numpy()
        self._s = self.test["score"].to_numpy()
        self._a = self.test["amt"].to_numpy()

    # ------------------------------------------------------------------ threshold & cost
    def policy_metrics(self, threshold: float, review_cost: float = REVIEW_COST,
                       friction_cost: float = FRICTION_COST) -> dict:
        m = confusion_at(self._y, self._s, self._a, threshold, review_cost, friction_cost)
        m["alerts_per_day"] = m["alerts"] / self.days
        m["no_model_cost"] = float(self._a[self._y == 1].sum())
        m["savings"] = m["no_model_cost"] - m["total_cost"]
        return m

    def cost_curve(self, review_cost=REVIEW_COST, friction_cost=FRICTION_COST, n=120) -> pd.DataFrame:
        grid = np.unique(np.concatenate([np.linspace(0.02, 0.99, n), [self.recommended]]))
        rows = [confusion_at(self._y, self._s, self._a, t, review_cost, friction_cost) for t in grid]
        df = pd.DataFrame(rows)
        df["alerts_per_day"] = df["alerts"] / self.days
        return df

    def best_threshold(self, review_cost=REVIEW_COST, friction_cost=FRICTION_COST) -> float:
        """Lowest-cost threshold for the given costs, chosen on VALIDATION data (never on test)."""
        y, s, a = self.val["is_fraud"].to_numpy(), self.val["score"].to_numpy(), self.val["amt"].to_numpy()
        grid = np.linspace(0.02, 0.99, 98)
        costs = [confusion_at(y, s, a, t, review_cost, friction_cost)["total_cost"] for t in grid]
        return float(grid[int(np.argmin(costs))])

    # ------------------------------------------------------------------ review queue
    def dates(self) -> list:
        return sorted(self.test["time"].dt.date.unique())

    def queue(self, date, threshold: float, limit: int = 40) -> pd.DataFrame:
        day = self.test[self.test["time"].dt.date == pd.Timestamp(date).date()]
        q = day[day["score"] >= threshold].sort_values("score", ascending=False).head(limit).copy()
        q["reasons"] = self.ex.reasons(q, top=3) if len(q) else []
        q["category_name"] = q["category"].astype(str).map(CATEGORY_NAMES)
        q["day_total"] = len(day)
        return q

    # ------------------------------------------------------------------ explanations
    def contributions(self, row: pd.DataFrame) -> pd.DataFrame:
        sv = self.ex.shap_values(row)[0]
        return (pd.DataFrame({"Feature": [FEATURE_LABELS.get(c, c) for c in self.ex.cols], "Impact": sv})
                .assign(abs=lambda d: d["Impact"].abs()).sort_values("abs", ascending=False).drop(columns="abs"))

    def global_importance(self, n: int = 6000) -> pd.Series:
        return self.ex.global_importance(self.test.sample(n, random_state=0))

    def what_if(self, amt: float, category: str, hour: int, amt_vs_cust_avg: float = 1.0,
                txns_last_24h: int = 2, amt_last_24h: float = 150.0, hours_since_prev: float = 6.0,
                age: float = 45.0, cust_prior_txns: int = 500, distance_km: float = 75.0,
                new_category_for_cust: int = 0) -> tuple[float, pd.DataFrame, list]:
        """Score a made-up transaction (for exploring the model, not for decisions)."""
        base = self.test[self.ex.cols].median(numeric_only=True)
        row = {c: base.get(c, 0.0) for c in self.ex.cols}
        row.update({"amt": amt, "log_amt": np.log1p(amt), "hour": hour, "is_night": int(hour >= 22 or hour < 4),
                    "amt_vs_cust_avg": amt_vs_cust_avg, "amt_vs_cust_cat_avg": amt_vs_cust_avg,
                    "txns_last_24h": txns_last_24h, "amt_last_24h": amt_last_24h,
                    "hours_since_prev": hours_since_prev, "age": age, "cust_prior_txns": cust_prior_txns,
                    "distance_km": distance_km, "new_category_for_cust": new_category_for_cust,
                    "category_risk": self.ex.cat_risk.get(category, 0.005)})
        df = pd.DataFrame([row])[self.ex.cols]
        score = float(self.ex.model.predict_proba(df)[0, 1])
        full = df.assign(category=category)
        return score, self.contributions(df), self.ex.reasons(full, top=4)[0]

    # ------------------------------------------------------------------ reports
    @staticmethod
    def model_comparison() -> pd.DataFrame:
        return pd.read_csv(REPORTS / "model_comparison.csv")

    @staticmethod
    def threshold_comparison() -> pd.DataFrame:
        return pd.read_csv(REPORTS / "threshold_comparison.csv")
