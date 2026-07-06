# HANDOFF.md — Amex Campus Challenge 2026

> **Last updated:** 2026-07-07  
> **Repo:** https://github.com/AnuragP004/amex (public)  
> **Project root:** `/home/thearagun/amex`

---

## Goal

Identify the **most profitable Premier Cardmembers** for American Express by building a profitability ranking model. The competition evaluates predictions via an accuracy/ranking metric against hidden ground truth. We must produce a `final_submission.xlsx` file containing:

1. A **'Predictions'** sheet with columns `ID` and `Prediction` (a continuous profitability score for each of 500k cardmembers).
2. A **'Profitability Framework'** sheet documenting the methodology across 10 required sections (Variables Used, Profitability Equation, Prediction Logic, Variable Selection Logic, Coefficient/Weight Derivation, Feature Transformations, Business Logic, Assumptions, Validation Approach, Additional Notes).

The submission template is `6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx`.

---

## Current Progress

### Accuracy Progression

| Version | Script | Accuracy | Status |
|---------|--------|----------|--------|
| V1 | `solve.py` | **0.305** | ❌ Superseded |
| V2 | `solve_v2.py` | **0.658** | ❌ Superseded |
| V3 | `solve_v3.py` | **TBD** | ✅ Current — awaiting submission result |

### What's Been Built

- **Data pipeline:** Missing value imputation (median for risk `f11`, mode for `f19`/`f20`, 0 for activity features), log transforms for 14 skewed financial features, 15 engineered features.
- **Domain profitability equation (P&L):** Revenue (annual fees, category-adjusted interchange at 2–3%, net interest margin at 15%, digital engagement proxy, collection recovery) minus Cost (PD×EAD×LGD credit loss, tiered rewards earning at $0.007/pt, benefit utilization, servicing).
- **Weak Supervision ML model:** `HistGradientBoostingClassifier` trained on pseudo-labels derived from domain score extremes (top 15% = profitable, bottom 50% = unprofitable). Learns non-linear feature interactions.
- **PCA composite:** 10-component PCA on 38 RobustScaled features, components aligned with domain score direction.
- **Tri-Ensemble:** 40% Domain Rank + 40% ML Rank + 20% PCA Rank, all using **percentile rank scaling** (not min-max).
- **Profitability Framework sheet:** All 10 sections populated with detailed methodology documentation.
- **Multi-agent brainstorming:** Three subagents (Skeptic, Constraint Guardian, User Advocate) reviewed V1 equation and identified 5 critical flaws, all resolved in V2+.

---

## What Worked

1. **Category-adjusted interchange rates** (airlines/lodging 3%, dining 2.5%, other 2%) instead of a flat 2.5% — significantly improved revenue signal accuracy.
2. **PD × EAD × LGD credit loss model** instead of naive PD × balance — properly accounts for transactional exposure and lending line exposure.
3. **Net interest margin (15%)** instead of gross APR (20%) — subtracting cost of funds was a critical fix from the Skeptic agent.
4. **Percentile rank scaling** instead of MinMaxScaler for ensembling — financial data has extreme outliers that compress 99% of scores near zero under min-max; rank scaling is completely immune.
5. **Weak supervision pseudo-labeling** — training a GBM on the clear extremes of the domain score lets it discover non-linear interactions (e.g., high revolve × high risk = loss; high revolve × low risk = profit) that a linear equation misses.
6. **Template preservation via `shutil.copy` + cell-level updates** — early attempts using `openpyxl` alone would lose the 'Profitability Framework' sheet. Copying the template first and writing into it solved this.
7. **`f11` median imputation** instead of 0 — filling risk with 0 would falsely mark missing-risk customers as zero-default-probability.

## What Didn't Work

1. **V1 pure rule-based equation** (0.305 accuracy) — flat 2.5% MDR, gross APR instead of net margin, no cost of funds, PD×balance instead of PD×EAD×LGD.
2. **PCA with only 3 components** — correlation with domain score was only 0.18; too much information loss.
3. **Isolation Forest for anomaly-based scoring** — negative correlation (-0.12) with domain score; anomaly detection doesn't map to profitability.
4. **MinMaxScaler for score blending** — extreme spenders (millions in `f5`) compressed all normal scores near 0, destroying ranking signal.
5. **`openpyxl` creating workbook from scratch** — lost the pre-existing 'Profitability Framework' sheet formatting and structure from the template.

---

## Key Files

| File | Purpose |
|------|---------|
| `solve_v3.py` | **Current best model.** Tri-Ensemble (Domain + ML + PCA) with rank scaling. Run this to regenerate `final_submission.xlsx`. |
| `solve_v2.py` | Previous model. Domain + PCA ensemble with min-max scaling (0.658). |
| `solve.py` | Original V1 model. Pure domain equation (0.305). |
| `fill_framework.py` | Standalone script to populate the 'Profitability Framework' tab. |
| `format_submission.py` | Standalone script to inject predictions into the template. |
| `final_submission.xlsx` | **Current submission file.** Generated by `solve_v3.py`. |
| `6a3eb196bc7a3_campus_challenge_r1_data.csv` | Source dataset (500k rows, 24 columns: id + f1–f23). |
| `6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx` | Official submission template (must preserve both sheets). |
| `6a3eb1af6b994_feature_description.csv` | Feature metadata. |
| `6a3ea743b1106_campus_challenge26_r1.pdf` | Problem statement PDF. |

---

## Data Summary

- **500,000 rows**, 24 columns (`id` + `f1`–`f23`)
- Key features: `f1` (avg revolve balance), `f5` (total spend 12m), `f6`–`f10` (category spend), `f11` (risk score, range 0–0.33), `f19` (supp accounts), `f20` (active charge cards), `f21` (rewards redeemed)
- `f23` has ~87% missing values (email clicks) — imputed to 0
- `f11` has significant missing values — imputed with median (~0.000643)

---

## Next Steps

1. **Submit V3 and check accuracy.** The user hasn't reported the V3 score yet. If it beats 0.658, iterate further; if not, debug.
2. **Tune the ensemble blend ratio.** Currently 40/40/20. Try 50/30/20 or 30/50/20 to see which signal dominates.
3. **Tune the pseudo-label thresholds.** Currently top 15% vs bottom 50%. Try top 20% vs bottom 40%, or top 10% vs bottom 60%.
4. **Hyperparameter tuning for GBM.** `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf` are all at defaults. Use a grid or Bayesian search.
5. **Try stacking instead of linear blending.** Train a logistic regression or another GBM on top of the three component scores.
6. **Add interaction features explicitly.** `f1 * f11` (revolve × risk), `f5 * f20` (spend per card), `benefit_intensity / f5` (cost-to-spend ratio).
7. **Investigate if XGBoost/LightGBM are available.** `HistGradientBoostingClassifier` is good but dedicated gradient boosting libraries may squeeze more signal.
8. **Consider target encoding for `f19`, `f20`** (discrete features with few unique values).

---

## Environment

- **Python 3** with `pandas`, `numpy`, `scikit-learn`, `openpyxl`
- **GitHub CLI** (`gh`) available and authenticated as `AnuragP004`
- **Git** configured with credential store
- **OS:** Arch Linux
