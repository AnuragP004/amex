# Amex Campus Challenge 2026 — Profitability Framework

## Overview
Solution for the **American Express Campus Challenge 2026 (Round 1)** — identifying the most profitable Premier Cardmembers using a data-driven Tri-Ensemble approach.

## Approach
We built a **three-stage ensemble** that blends domain expertise with machine learning:

| Component | Weight | Method |
|---|---|---|
| **Domain Score** | 40% | Hand-crafted P&L equation (Revenue − Cost) using industry heuristics |
| **ML Score** | 40% | `HistGradientBoostingClassifier` trained via Weak Supervision on domain extremes |
| **PCA Score** | 20% | 10-component PCA composite weighted by explained variance |

All scores are blended using **Percentile Rank scaling** (immune to financial outliers).

## Key Files
| File | Description |
|---|---|
| `solve_v3.py` | **Final submission script** — Tri-Ensemble with Weak Supervision |
| `solve_v2.py` | V2 script — Domain + PCA ensemble (0.658 accuracy) |
| `solve.py` | V1 script — Pure domain equation (0.305 accuracy) |
| `fill_framework.py` | Populates the 'Profitability Framework' Excel tab |
| `format_submission.py` | Injects predictions into the submission template |
| `6a3eb1af6b994_feature_description.csv` | Feature metadata (23 variables) |

## Revenue Model
- **Annual Fees**: $625/primary card + $175/supplementary
- **Interchange**: Category-adjusted MDRs (Airlines/Lodging 3.0%, Dining 2.5%, Other 2.0%)
- **Net Interest**: 15% margin on revolving balances (20% APR − 5% CoF)

## Cost Model
- **Credit Loss**: PD × EAD × LGD (40%) framework
- **Rewards**: Tiered earn rates (1x–5x) at $0.007/point blended cost
- **Benefits**: Lounge ($30/visit), Airline/Cab/Entertainment credits
- **Servicing**: Cancellation ($50/call) and Collection ($40/call) costs

## Accuracy Progression
| Version | Accuracy | Key Change |
|---|---|---|
| V1 | 0.305 | Pure rule-based domain equation |
| V2 | 0.658 | + PCA ensemble + category-adjusted interchange + PD×EAD×LGD |
| V3 | TBD | + Weak Supervision ML + Percentile Rank ensembling |

## Requirements
```
pandas
numpy
scikit-learn
openpyxl
```
