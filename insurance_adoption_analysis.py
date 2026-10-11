pip install pandas numpy scipy statsmodels pingouin factor_analyzer

"""
A Study on Adoption of Online Vehicle Insurance in Bengaluru
Replication and Analytical Pipeline Code

This script provides the end-to-end analytical pipeline for the study, including:
1. Data Preprocessing & Validation
2. Harman's Single Factor Test for Common Method Bias (CMB)
3. Exploratory Factor Analysis (PCA)
4. Construct Reliability (Cronbach's Alpha)
5. Multicollinearity Diagnostics (VIF)
6. Bivariate Independence Tests (Chi-Square & Fisher's Exact)
7. Econometric Modelling (Logistic Regression & Linear Probability Model)

Author: [Divyam Kumar]
License: MIT
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import warnings

# Suppress warnings for cleaner output in production
warnings.filterwarnings('ignore')

# --- CONFIGURATION & MAPPINGS ---

# Raw column names mapped to shorter variable names for clean code
# Note: The mapping keys are exactly as they appear in the survey, 
# but we will strip whitespace dynamically during loading to ensure perfect matches.
COLUMN_MAPPING = {
    'Do you currently own & maintain insurance for a motorized vehicle in Bangalore?': 'Owns_Vehicle',
    'Your Age Group': 'Age_Group',
    'How did you purchase or renew your vehicle insurance most recently?': 'Recent_Channel_Raw',
    'For my next vehicle insurance renewal, I intend to:': 'Future_Renewal_Intent',
    
    # Core Constructs
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Online platforms allow me to compare quotes and issue policies faster than offline agents.]': 'PE1',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Instant soft-copy policy issuance via email/WhatsApp is highly convenient.]': 'PE2',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Navigating online insurance apps/websites and understanding coverage options is easy.]': 'EE1',
    'Performance Expectancy (PE) & Effort Expectancy (EE)   [Uploading inspection photos/videos online during renewal requires minimal effort.]': 'EE2',
    'Price Value (PV) & Social Influence (SI)   [Online platforms offer lower premiums and direct discounts due to zero agent commission.]': 'PV1',
    'Price Value (PV) & Social Influence (SI)   [Add-on covers (Zero Dep, Engine Protect) are transparently priced on digital platforms.]': 'PV2',
    'Price Value (PV) & Social Influence (SI)   [Recommendations from colleagues/friends in Bangalore influence my choice to buy online.]': 'SI1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I worry about lack of personal assistance during physical claim settlement without an agent.]': 'PR1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I feel secure entering my financial and vehicle details on online insurance portals.]': 'PR2',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [I trust digital-first insurers (e.g., Acko/Digit) as much as traditional state/private banks.]': 'PT1',
    'Perceived Risk (PR), Trust (PT), and Claim Experience   [Automated digital claim processing (self-inspection via mobile app) is reliable.]': 'PT2',
    
    # Emerging Trends (Used for Harman's Single Factor Test)
    'Perception Toward Emerging Trends   [I would prefer a "Pay-As-You-Drive" policy where premium depends on kilometers driven.]': 'ET1',
    'Perception Toward Emerging Trends   [Mandatory Digilocker integration and IRDAI e-Insurance accounts make online policies easier to manage.]': 'ET2',
    'Perception Toward Emerging Trends   [I am comfortable sharing driving data via telematics/app if it gives me lower premiums.]': 'ET3'
}

LIKERT_DICT = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

DIGITAL_CHANNELS = [
    'Online Aggregator (e.g., PolicyBazaar)',
    'Bank Portal / UPI App (e.g., PhonePe, YONO SBI)',
    'Traditional Insurer Web Portal / App (e.g., Bajaj Allianz)'
]

ALL_LIKERT_COLS = ['PE1', 'PE2', 'EE1', 'EE2', 'PV1', 'PV2', 'SI1', 
                   'PR1', 'PR2', 'PT1', 'PT2', 'ET1', 'ET2', 'ET3']


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Loads dataset, cleans column names, filters validated sample, and engineers features."""
    df = pd.read_csv(filepath)
    
    # Strip whitespace from raw columns and our mapping keys to prevent KeyErrors
    df.columns = df.columns.str.strip()
    clean_mapping = {k.strip(): v for k, v in COLUMN_MAPPING.items()}
    df = df.rename(columns=clean_mapping)
    
    # 1. Filter Validated Sample
    df = df[df['Owns_Vehicle'].str.strip().str.lower() == 'yes'].copy()
    print(f"[*] Validated sample size: N = {len(df)}")
    
    # 2. Map Likert scales to numeric
    for col in ALL_LIKERT_COLS:
        if col in df.columns:
            df[col] = df[col].map(LIKERT_DICT)
            
    # 3. Engineer Outcome Variables
    df['Digital_Intent'] = df['Future_Renewal_Intent'].apply(
        lambda x: 1 if x in ['Continue using an Online Channel', 
                             'Switch completely to an Online Channel (Aggregator / Direct App)'] else 0
    )
    
    df['Recent_Channel'] = df['Recent_Channel_Raw'].apply(
        lambda x: 'Digital' if x in DIGITAL_CHANNELS else 'Offline'
    )
    
    # 4. Compute Composite Scores (following exact original logic)
    df['PE_score'] = df['PE1']
    df['EE_score'] = df[['EE1', 'EE2']].mean(axis=1)
    df['PV_score'] = df[['PV1', 'PV2']].mean(axis=1)
    df['PT_score'] = df[['PT1', 'PT2', 'PR2']].mean(axis=1)  # PR2 merged into PT based on PCA
    df['PR_score'] = df['PR1']
    df['SI_score'] = df['SI1']
    
    return df


def test_common_method_bias(df: pd.DataFrame):
    """Executes Harman's Single Factor Test for Common Method Bias."""
    print("\n--- 1. HARMAN'S SINGLE FACTOR TEST (CMB) ---")
    
    X = df[ALL_LIKERT_COLS].dropna()
    pca = PCA()
    pca.fit(X)
    
    variance_first_factor = pca.explained_variance_ratio_[0] * 100
    print(f"Total number of items tested: {X.shape[1]}")
    print(f"Variance explained by a single factor: {variance_first_factor:.2f}%")
    
    if variance_first_factor < 50:
        print("CONCLUSION: Because the single factor explains < 50% of variance, CMB is NOT a significant issue.")
    else:
        print("CONCLUSION: Because the single factor explains >= 50% of variance, CMB IS a significant issue.")


def exploratory_factor_analysis(df: pd.DataFrame):
    """Validates structural constructs using PCA."""
    print("\n--- 2. PRINCIPAL COMPONENT ANALYSIS (FACTOR VALIDATION) ---")
    pca_cols = ['PE1', 'PE2', 'EE1', 'EE2', 'PV1', 'PV2', 'PR1', 'PR2', 'PT1', 'PT2']
    X_pca = df[pca_cols].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=4, random_state=42)
    pca.fit(X_scaled)
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X_pca.columns)
    print("Factor Loadings:")
    print(loadings.round(3))


def evaluate_reliability(df: pd.DataFrame):
    """Computes Cronbach's Alpha for multi-item constructs."""
    print("\n--- 3. CONSTRUCT RELIABILITY (CRONBACH'S ALPHA) ---")
    ee_alpha = pg.cronbach_alpha(data=df[['EE1', 'EE2']])[0]
    pv_alpha = pg.cronbach_alpha(data=df[['PV1', 'PV2']])[0]
    pt_alpha = pg.cronbach_alpha(data=df[['PT1', 'PT2', 'PR2']])[0]
    
    print(f"EE Alpha: {ee_alpha:.3f}")
    print(f"PV Alpha: {pv_alpha:.3f}")
    print(f"PT (Revised with PR2) Alpha: {pt_alpha:.3f}")


def check_multicollinearity(df: pd.DataFrame, predictors: list):
    """Calculates Variance Inflation Factor (VIF) for the model predictors."""
    print("\n--- 4. MULTICOLLINEARITY (VIF) ---")
    X = df[predictors].dropna()
    X_with_const = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    print(vif_data.round(3))


def test_bivariate_independence(df: pd.DataFrame):
    """Runs Chi-Square and Fisher's Exact tests on Demographics vs Channel."""
    print("\n--- 5. BIVARIATE INDEPENDENCE (AGE VS CHANNEL) ---")
    
    # Pearson Chi-Square
    contingency_table = pd.crosstab(df['Age_Group'], df['Recent_Channel'])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Pearson Chi-Square: {chi2:.4f}, p-value: {p_chi2:.4f}, df: {dof}")
    
    # Fisher's Exact Test (Aggregated Age)
    df['Age_Binary'] = df['Age_Group'].apply(lambda x: '<=40' if x in ['18 – 25 years', '26 – 40 years'] else '>40')
    fisher_table = pd.crosstab(df['Age_Binary'], df['Recent_Channel'])
    
    if fisher_table.shape == (2, 2):
        odds_ratio, p_fisher = stats.fisher_exact(fisher_table)
        print(f"Fisher's Exact Test (<=40 vs >40): OR = {odds_ratio:.4f}, p-value = {p_fisher:.4e}")
    else:
        print("Fisher's Exact Test skipped: Table is not 2x2.")


def run_econometric_models(df: pd.DataFrame, predictors: list):
    """Executes Logistic Regression and Linear Probability Models."""
    print("\n--- 6. ECONOMETRIC MODELLING ---")
    
    model_data = df[['Digital_Intent'] + predictors].dropna()
    y = model_data['Digital_Intent']
    X = sm.add_constant(model_data[predictors])
    
    print("\n[ LOGISTIC REGRESSION ]")
    logit_model = sm.Logit(y, X).fit(disp=0)
    print(logit_model.summary())
    
    print("\n[ LINEAR PROBABILITY MODEL (HC3 ROBUST SE) ]")
    lpm_model = sm.OLS(y, X).fit(cov_type='HC3')
    print(lpm_model.summary())


def main():
    """Main execution pipeline."""
    # Define dataset path
    DATA_FILE = "survey_data_anonymized.csv"
    
    try:
        # Load and process data
        df = load_and_preprocess(DATA_FILE)
        
        # Run analytical steps
        test_common_method_bias(df)
        exploratory_factor_analysis(df)
        evaluate_reliability(df)
        
        # Predictors for Econometric Models
        predictors = ['PE_score', 'EE_score', 'PV_score', 'PR_score', 'PT_score', 'SI_score']
        
        check_multicollinearity(df, predictors)
        test_bivariate_independence(df)
        run_econometric_models(df, predictors)
        
        print("\n[*] Pipeline execution completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found. Please ensure it is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()