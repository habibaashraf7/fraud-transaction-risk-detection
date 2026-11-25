# Memo: Recommended Alert Threshold for Card Transaction Risk Review

**To:** Head of Fraud Operations · **From:** ML intern (UneeQ Interns) · **Re:** fraud risk-score threshold

---

## Recommendation

**Send every transaction with a risk score ≥ 0.76 to the analyst review queue.**
Review the threshold **quarterly**, and whenever fraud patterns or team costs change.

**The model never blocks or declines a payment.** Transactions above the threshold are *queued for a human analyst*, who decides whether to confirm fraud (contact the customer and dispute), release, or escalate. Transactions below the threshold proceed normally.

## Expected impact (back-test: 21 Jun – 31 Dec 2020, 194 days, 555,719 transactions)

| | No model | **Threshold 0.76** |
|---|---|---|
| Alerts per day | 0 | **≈ 12** (≈ 2 analyst-hours/day at 10 min each) |
| Share of alerts that are real fraud (precision) | — | **88%** |
| Share of fraud caught (recall) | 0% | **94%** (2,017 of 2,145 frauds) |
| Fraud $ caught | $0 | **$1.10M** of $1.13M |
| Fraud $ missed | $1.13M | $29.5K |
| Total cost (reviews + friction + missed fraud) | $1.13M | **$66.8K** |
| **Savings** | — | **≈ $1.07M over 6 months** |

## Why 0.76 and not another threshold?

The threshold was chosen on **validation data (Mar–Jun 2020)**, then checked once on the later test period.

| Policy | Alerts/day | Precision | Recall | Total cost |
|---|---|---|---|---|
| Default 0.5 | 13.6 | 78% | 96% | $63.3K |
| **Lowest cost on validation (0.76)** | **11.9** | **88%** | **94%** | **$66.8K** |
| Best F1 score (0.89) | 10.9 | 93% | 92% | $71.9K |
| Fill team capacity (120/day) | 107 | 10% | 99.5% | $500.6K |

* **The cost curve is flat between about 0.5 and 0.85.** On the test period, 0.5 was about $3.5K cheaper (0.3%), which is within normal variation. We prefer **0.76** because it was chosen on independent data and keeps precision high (88%), so analysts see fewer false alarms and customers are bothered less.
* **Best F1 isn't best for the business.** It treats a missed $900 fraud the same as a false alarm on a $5 purchase, so it misses more expensive fraud.
* **More alerts ≠ more protection.** Filling the team's capacity adds about 95 false alarms a day for almost no extra fraud caught. Total cost rises 7×.

## Cost assumptions and sensitivity

| Item | Assumption |
|---|---|
| Analyst review | **$15** per alert (about 10 minutes) |
| Customer friction | **$10** per false alarm (a call, or a temporary hold) |
| Missed fraud | its **full amount**, charged back to the bank |

The recommendation is **stable**: the best threshold stays at **0.76** for review costs of $5–50 and friction costs of $0–50.

| If… | Then adjust to… |
|---|---|
| Analyst time costs **≥ $50–100 per alert** | ≈ **0.90** (fewer, higher-precision alerts) |
| Reviews are **almost free** ($5, no friction) | ≈ **0.38** (more alerts, higher recall) |

The live console (`streamlit run app.py`) lets operations re-run this trade-off with their own cost figures.

## Safeguards and conditions

1. **Human decision.** Analysts own every outcome. The score and its top 3 reasons (e.g. *"$3,060 spent in the previous 24h"*, *"5.2× this customer's usual amount"*) support, not replace, judgement.
2. **Fairness.** Gender is excluded from the model. Recall (92–98%) and false-alarm rates (0.04–0.06 per 100 legit transactions) are similar across age groups. Re-check this at every review.
3. **Monitoring.** Track weekly precision, recall, alert volume and the score distribution. Retrain if precision drops below 75% or alert volume changes by more than 30%.
4. **Pilot first.** These results come from **simulated** data. Run 4–8 weeks in shadow mode on real transactions, scoring without alerting, before going live.
