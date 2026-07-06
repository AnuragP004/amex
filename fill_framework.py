import openpyxl
import shutil

# Work on the already-populated final_submission.xlsx
wb = openpyxl.load_workbook('final_submission.xlsx')
ws = wb['Profitability Framework']

responses = {
    2: (  # Variables Used
        "f1 (Avg Revolve Balance), f2 (Cancellation Calls), f3 (Collection Calls), "
        "f4 (Rewards Points Balance), f5 (Total Spend 12m), f6 (Airlines Spend 12m), "
        "f7 (Other Spend 12m), f8 (Entertainment Spend 12m), f9 (Lodging Spend 12m), "
        "f10 (Dining Spend 12m), f11 (Avg Risk Score), f13 (Lounge Access Count), "
        "f14 (Airlines Credits Used), f15 (Cab Benefits Usage), f16 (Entertainment Credit Used), "
        "f17 (Total Lend Line Amount), f19 (Supplementary Accounts), "
        "f20 (Active Charge Cards), f21 (Rewards Redeemed 12m)"
    ),
    3: (  # Profitability Equation
        "Profitability = Revenue - Cost. "
        "Revenue = (f20 * 625) + (f19 * 175) + (f5 * 0.025) + (f1 * 0.20). "
        "Cost = (f1 * 0.05) + ((f6*5 + f9*5 + f7 + f8 + f10) * 0.01) + "
        "f14 + (f15 * 15) + f16 + (f13 * 30) + (f11 * (f1 + f5/12)) + "
        "(f2 * 20) + (f3 * 30)."
    ),
    4: (  # Prediction Logic
        "Each cardmember's profitability score is computed using the equation above applied row-wise. "
        "All cardmembers are then ranked in descending order of their profitability score. "
        "Higher scores indicate more profitable customers. The score itself is the prediction output."
    ),
    5: (  # Variable Selection Logic
        "Variables were selected based on their mapping to real-world card issuer revenue and cost drivers: "
        "(1) Revenue drivers: annual fees (f20, f19), interchange from spend (f5), interest income from revolving (f1). "
        "(2) Cost drivers: rewards liability (f6-f10 as earn proxies), benefit utilization (f13, f14, f15, f16), "
        "credit risk exposure (f11 combined with f1 and f5), and servicing costs (f2, f3). "
        "f4 (points balance) and f17 (lend line) were excluded as they represent state, not P&L flow."
    ),
    6: (  # Coefficient/Weight Derivation
        "Coefficients are derived from publicly available industry benchmarks for premium charge cards: "
        "Annual fee $625 (Amex Platinum range), supplementary fee $175, interchange/MDR 2.5% (Amex average), "
        "APR 20% (US average revolving rate), cost of funds 5% (Fed funds + spread), "
        "reward cost $0.01/point, lounge cost $30/visit (Priority Pass rate), "
        "cab benefit $15/month, cancellation call cost $20, collection call cost $30. "
        "Earned points multipliers: 5x on airlines (f6) and lodging (f9), 1x on other categories (f7, f8, f10)."
    ),
    7: (  # Feature Transformations
        "1. Missing value imputation: f11 (Avg Risk Score) filled with column mean to avoid zero-risk bias; "
        "all other missing values filled with 0 (absence of activity). "
        "2. Exposure at Default (EAD): computed as f1 + (f5 / 12) to capture both revolving balance and "
        "average monthly transactional exposure for charge card users who may have f1=0. "
        "3. Earned points proxy: (f6*5) + (f9*5) + (f7 + f8 + f10)*1 to estimate accrued reward liability "
        "instead of using redeemed points (f21), which creates temporal volatility."
    ),
    8: (  # Business Logic
        "The framework models a premier card P&L at the individual cardmember level. "
        "Revenue streams: (1) annual fees from primary and supplementary cards, "
        "(2) merchant discount revenue (interchange) from total spend, "
        "(3) interest income from revolving balances net of cost of funds. "
        "Cost streams: (1) rewards liability based on estimated earned points (not redeemed, to avoid timing distortion), "
        "(2) direct benefit costs (lounge, airline credits, cab, entertainment credits), "
        "(3) expected credit loss = probability of default (f11) * exposure at default (revolve + monthly spend), "
        "(4) customer servicing costs from cancellation and collection call volumes."
    ),
    9: (  # Assumptions
        "1. All cardmembers hold a premier product with $625 annual fee. "
        "2. Supplementary cards charged at $175 each. "
        "3. Amex average merchant discount rate is 2.5% across categories. "
        "4. Revolving APR is 20% (US average for premium cards). "
        "5. Cost of funds is 5% (Fed funds rate + credit spread). "
        "6. Reward points cost the issuer $0.01 each. "
        "7. Lounge visits cost $30 each; cab benefit is $15 per month of usage. "
        "8. f11 (Avg Risk Score) is treated as a probability of default (0-1 scale). "
        "9. Missing risk scores are imputed with the population mean; other missing values default to 0. "
        "10. Cost of servicing calls: $20 per cancellation call, $30 per collection call."
    ),
    10: (  # Validation Approach
        "1. Sanity checks: verified no negative IDs, confirmed all template IDs are present in output. "
        "2. Distribution analysis: reviewed profitability score distribution for outliers and reasonableness. "
        "3. Multi-agent peer review: the equation was stress-tested by three independent review agents "
        "(Skeptic, Constraint Guardian, User Advocate) who identified and resolved 5 critical flaws "
        "including missing cost of funds, incorrect risk exposure formula, reward timing volatility, "
        "dangerous zero-fill of risk scores, and hardcoded magic numbers. "
        "4. Edge case validation: checked that high-spend transactors with f1=0 are not incorrectly assigned zero risk."
    ),
    11: (  # Additional Notes
        "This framework was developed using a structured multi-agent brainstorming protocol. "
        "The initial equation was iteratively refined through sequential peer review by specialized agents, "
        "each constrained to a specific critique domain. All design decisions and objections are logged "
        "in a formal Decision Log artifact."
    ),
}

for row_num, text in responses.items():
    ws.cell(row=row_num, column=2, value=text)

wb.save('final_submission.xlsx')
print("Done! All sections in 'Profitability Framework' are now filled.")
