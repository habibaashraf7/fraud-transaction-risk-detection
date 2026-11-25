"""Explain risk scores with SHAP and turn them into plain-English reason codes for analysts.

SHAP splits each transaction's score into one contribution per feature
(positive = pushes towards fraud, negative = pushes towards legit). The contributions add up
exactly to the model's raw score, so the reasons always match what the model actually did.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from .features import FEATURE_LABELS
from .train import MODELS

FEATURE_LABELS = {**FEATURE_LABELS, "category_risk": "Merchant category risk"}
CATEGORY_NAMES = {"shopping_net": "online shopping", "misc_net": "misc. online", "grocery_pos": "grocery (in store)",
                  "shopping_pos": "shopping (in store)", "gas_transport": "gas & transport", "misc_pos": "misc. in store",
                  "grocery_net": "online grocery", "travel": "travel", "entertainment": "entertainment",
                  "personal_care": "personal care", "kids_pets": "kids & pets", "food_dining": "food & dining",
                  "home": "home", "health_fitness": "health & fitness"}


class Explainer:
    def __init__(self):
        info = json.loads((MODELS / "model_info.json").read_text())
        self.model = joblib.load(MODELS / "fraud_model.joblib")
        self.cols = info["features"]
        self.cat_risk = info["category_risk"]
        self.tree = shap.TreeExplainer(self.model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return self.tree.shap_values(X[self.cols])

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        sv = self.shap_values(X)
        return (pd.Series(np.abs(sv).mean(0), index=[FEATURE_LABELS.get(c, c) for c in self.cols])
                .sort_values(ascending=False))

    # ------------------------------------------------------------------ reason codes
    def _phrase(self, f: str, r: pd.Series) -> str:
        v = r[f]
        if f in ("amt", "log_amt"):
            if r["amt"] >= 200:
                return f"Large amount: ${r['amt']:,.2f}"
            if r["amt"] < 30:
                return f"Small amount (${r['amt']:,.2f}), a typical card-testing pattern"
            return f"Amount ${r['amt']:,.2f}"
        if f == "amt_vs_cust_avg":
            return f"{v:.1f}× this customer's usual amount" if v >= 1 else f"Below this customer's usual amount ({v:.1f}×)"
        if f == "amt_vs_cust_cat_avg":
            return f"{v:.1f}× their usual spend in this category"
        if f == "category_risk":
            name = CATEGORY_NAMES.get(str(r["category"]), str(r["category"]))
            if v >= 0.007:
                return f"High-risk category: {name} ({v:.1%} historic fraud rate)"
            return f"Unusual pattern for {name} purchases"
        if f in ("hour", "is_night"):
            h = int(r["hour"])
            return f"At {h % 12 or 12} {'AM' if h < 12 else 'PM'}{' (night-time)' if r['is_night'] else ''}"
        if f == "amt_last_24h":
            return f"${v:,.0f} spent in the previous 24h"
        if f == "txns_last_24h":
            return f"{int(v)} other purchases in the previous 24h"
        if f == "hours_since_prev":
            return (f"Only {v * 60:.0f} min after the previous purchase" if v < 1
                    else f"{v:.1f} h since the previous purchase")
        if f == "new_category_for_cust":
            return "First-ever purchase in this category" if v else "Customer has shopped in this category before"
        if f == "cust_prior_txns":
            return (f"Short history: only {int(v)} earlier transactions" if v < 30
                    else f"Customer history: {int(v)} earlier transactions")
        if f == "age":
            return f"Customer age {v:.0f}"
        if f == "distance_km":
            return f"Merchant {v:,.0f} km from home"
        if f == "day_of_week":
            return f"On a {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(v)]}"
        if f == "log_city_pop":
            return f"Home city population ~{10 ** v:,.0f}"
        return FEATURE_LABELS.get(f, f)

    def reasons(self, rows: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Top `top` features pushing each transaction TOWARDS fraud, as readable phrases."""
        sv = self.shap_values(rows)
        out = []
        for i, (_, r) in enumerate(rows.iterrows()):
            order = np.argsort(-sv[i])
            items, seen = [], set()
            for j in order:
                f = self.cols[j]
                if sv[i, j] <= 0 or len(items) == top:
                    break
                key = "amt" if f in ("amt", "log_amt") else "hour" if f in ("hour", "is_night") else f
                if key in seen:
                    continue
                seen.add(key)
                items.append({"feature": FEATURE_LABELS.get(f, f), "text": self._phrase(f, r),
                              "impact": float(sv[i, j])})
            out.append(items)
        return out
