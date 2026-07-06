import pandas as pd
import numpy as np

# Constants for Profitability Equation based on Multi-Agent Brainstorming
ANNUAL_FEE_PRIMARY = 625
ANNUAL_FEE_SUPP = 175
INTERCHANGE_RATE = 0.025
APR = 0.20
COST_OF_FUNDS = 0.05
REWARD_COST_PER_POINT = 0.01
CAB_CREDIT_PER_MONTH = 15
LOUNGE_COST_PER_VISIT = 30
CANCEL_CALL_COST = 20
COLLECTION_CALL_COST = 30

def calculate_profitability(df):
    # Impute missing values
    mean_risk_score = df['f11'].mean()
    df['f11'] = df['f11'].fillna(mean_risk_score)
    df = df.fillna(0)

    # REVENUES
    rev_fee = (df['f20'] * ANNUAL_FEE_PRIMARY) + (df['f19'] * ANNUAL_FEE_SUPP)
    rev_interchange = df['f5'] * INTERCHANGE_RATE
    rev_interest = df['f1'] * APR
    total_revenue = rev_fee + rev_interchange + rev_interest

    # COSTS
    cost_of_funds = df['f1'] * COST_OF_FUNDS

    # Points earned on: Airlines (f6, 5x), Lodging (f9, 5x), Other/Entertainment/Dining (f7+f8+f10, 1x)
    earned_points = (df['f6'] * 5) + (df['f9'] * 5) + (df['f7'] + df['f8'] + df['f10']) * 1
    cost_rewards = earned_points * REWARD_COST_PER_POINT

    cost_benefits = df['f14'] + (df['f15'] * CAB_CREDIT_PER_MONTH) + df['f16'] + (df['f13'] * LOUNGE_COST_PER_VISIT)

    # Exposure at Default = Revolve Balance + Average Monthly Spend
    exposure = df['f1'] + (df['f5'] / 12.0)
    cost_risk = df['f11'] * exposure

    cost_service = (df['f2'] * CANCEL_CALL_COST) + (df['f3'] * COLLECTION_CALL_COST)

    total_cost = cost_of_funds + cost_rewards + cost_benefits + cost_risk + cost_service

    # PROFITABILITY
    df['profitability_score'] = total_revenue - total_cost
    return df

def main():
    print("Loading data...")
    df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv')
    
    print("Calculating profitability...")
    df = calculate_profitability(df)
    
    print("Ranking customers...")
    df_sorted = df.sort_values(by='profitability_score', ascending=False)
    
    print("Loading submission template...")
    template = pd.read_excel('6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx')
    
    print("Template columns:", template.columns.tolist())
    
    # We expect the template to have 'id' and perhaps another column like 'probability' or 'rank'
    if 'id' in template.columns:
        # Match template rows order or just save top ids?
        # The prompt says: "rank order the cardmembers based on estimated profitability".
        # It's better to just output exactly in the order of the template but with our scores,
        # OR order by score descending. Let's see what the template looks like.
        
        # Merge with template to ensure we have exactly what they want
        submission = template[['id']].copy()
        
        # Merge with our scores
        df_scores = df[['id', 'profitability_score']]
        submission = submission.merge(df_scores, on='id', how='left')
        
        for col in template.columns:
            if col != 'id':
                # Replace whatever target column is in the template with our score
                submission[col] = submission['profitability_score']
                
        # Drop the intermediate column if it wasn't in the template
        if 'profitability_score' not in template.columns:
            submission = submission.drop(columns=['profitability_score'])
            
        submission.to_csv('submission.csv', index=False)
        print("Generated submission.csv")
    else:
        print("Warning: Template does not have 'id' column.")
        df_sorted[['id', 'profitability_score']].to_csv('submission.csv', index=False)
        print("Generated submission.csv")

if __name__ == '__main__':
    main()
