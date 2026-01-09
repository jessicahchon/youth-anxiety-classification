"""
NSCH 2023 Data Preprocessing
Project: Detection of Adolescent Anxiety Using Machine Learning
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# 1. Load Data
df_original = pd.read_excel("nsch_2023_topical.xlsx")
print(f"Original dataset shape: {df_original.shape}")


# 2. Filter to Adolescent Population (Ages 12-17)
df = df_original[
    (df_original['SC_AGE_YEARS'] >= 12) & 
    (df_original['SC_AGE_YEARS'] <= 17)
].copy()

print(f"After filtering ages 12-17: {df.shape}")
print(f"\nAge distribution:")
print(df['SC_AGE_YEARS'].value_counts().sort_index())


# 3. Remove Columns with High Missing Values (>30%)
missing_pct = df.isnull().sum() / len(df)
high_missing = missing_pct[missing_pct > 0.3].index.tolist()

print(f"\nColumns with >30% missing: {len(high_missing)}")
df = df.drop(columns=high_missing)
print(f"After removing high missing columns: {df.shape}")


# 4. Prepare Target Variable
# K2Q33A: "Has a doctor ever told you that this child has Anxiety Problems?"
# 1 = Yes, 2 = No
df = df[df['K2Q33A'].isin([1, 2])].copy()
df['TARGET'] = (df['K2Q33A'] == 1).astype(int)

print(f"\nAfter removing missing K2Q33A: {df.shape}")
print(f"TARGET distribution:\n{df['TARGET'].value_counts()}")
print(f"Prevalence: {df['TARGET'].mean()*100:.1f}%")


# 5. Remove Data Leakage Variables
remove_vars = [
    # Operational/Administrative
    'YEAR', 'CBSAFP_YN', 'FIPSST', 'FORMTYPE', 'FWC', 
    'HHID', 'METRO_YN', 'MPC_YN', 'STRATUM', 'TENURE_IF',
    
    # Target original and follow-up
    'K2Q33A', 'K2Q33B', 'K2Q33C',
    
    # Depression (high comorbidity - bidirectional leakage)
    'K2Q32A', 'K2Q32B', 'K2Q32C',
    
    # Mental health treatment (consequence of diagnosis)
    'K4Q22_R', 'K4Q23', 'K4Q27', 'K4Q28X04',
    
    # CSHCN related (composite includes mental health)
    'SC_CSHCN', 'SC_K2Q13', 'SC_K2Q14', 'SC_K2Q15',
    'SC_K2Q22', 'SC_K2Q23', 'TOTCSHCN', 'TOTNONSHCN',
    
    # Prescription medication related
    'SC_K2Q10', 'SC_K2Q11', 'SC_K2Q12',
    
    # Limited ability (downstream consequence)
    'HCABILITY', 'SC_K2Q16', 'SC_K2Q17', 'SC_K2Q18',
    
    # Other leakage
    'MEMORYCOND', 'C4Q04', 'ARRANGEHC',

    # Survey/Admin
    'HHLANGUAGE', 'BIRTH_MO', 'BIRTH_YR',
    
    # Derived age variables (redundant)
    'TOTAGE_0_5', 'TOTAGE_6_11', 'TOTAGE_12_17',
    'SC_AGE_LT4', 'SC_AGE_LT6', 'SC_AGE_LT9', 'SC_AGE_LT10',
    
    # Imputation flags
    'SC_RACE_R_IF', 'SC_HISPANIC_R_IF', 'SC_SEX_IF',
    'FPL_IF', 'A1_GRADE_IF', 'BIRTH_YR_F', 'HHCOUNT_IF',
]

df = df.drop(columns=[c for c in remove_vars if c in df.columns])
print(f"\nAfter removing leakage variables: {df.shape}")


# 6. Handle Missing Values (Mode Imputation)
missing_cols = df.columns[df.isnull().any()]
print(f"\nColumns with missing values: {len(missing_cols)}")

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"After imputation: {df.shape}")
print(f"Remaining missing: {df.isnull().sum().sum()}")


# 7. Validate No Data Leakage (Cramér's V)
def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = len(x)
    min_dim = min(contingency.shape) - 1
    return 0 if min_dim == 0 else np.sqrt(chi2 / (n * min_dim))

results = {}
for col in df.columns:
    if col != 'TARGET':
        try:
            mask = df[col].notna() & df['TARGET'].notna()
            if mask.sum() > 0:
                results[col] = cramers_v(df.loc[mask, col], df.loc[mask, 'TARGET'])
        except:
            pass

cramers_df = pd.DataFrame({
    'Variable': results.keys(),
    'Cramers_V': results.values()
}).sort_values('Cramers_V', ascending=False)

print("\nTop 30 correlations with TARGET:")
print(cramers_df.head(30).to_string(index=False))

high_corr = cramers_df[cramers_df['Cramers_V'] > 0.5]
print(f"\nVariables with Cramér's V > 0.5: {len(high_corr)}")


# 8. Save Preprocessed Data
df.to_excel("anxiety_preprocessed_FINAL.xlsx", index=False)

print(f"\nFinal shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Anxiety prevalence: {df['TARGET'].mean()*100:.1f}%")"""
NSCH 2023 Data Preprocessing
Project: Predicting Adolescent Anxiety Using Machine Learning
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# 1. Load Data
df_original = pd.read_excel("nsch_2023_topical.xlsx")
print(f"Original dataset shape: {df_original.shape}")


# 2. Filter to Adolescent Population (Ages 12-17)
df = df_original[
    (df_original['SC_AGE_YEARS'] >= 12) & 
    (df_original['SC_AGE_YEARS'] <= 17)
].copy()

print(f"After filtering ages 12-17: {df.shape}")
print(f"\nAge distribution:")
print(df['SC_AGE_YEARS'].value_counts().sort_index())


# 3. Remove Columns with High Missing Values (>30%)
missing_pct = df.isnull().sum() / len(df)
high_missing = missing_pct[missing_pct > 0.3].index.tolist()

print(f"\nColumns with >30% missing: {len(high_missing)}")
df = df.drop(columns=high_missing)
print(f"After removing high missing columns: {df.shape}")


# 4. Prepare Target Variable
# K2Q33A: "Has a doctor ever told you that this child has Anxiety Problems?"
# 1 = Yes, 2 = No
df = df[df['K2Q33A'].isin([1, 2])].copy()
df['TARGET'] = (df['K2Q33A'] == 1).astype(int)

print(f"\nAfter removing missing K2Q33A: {df.shape}")
print(f"TARGET distribution:\n{df['TARGET'].value_counts()}")
print(f"Prevalence: {df['TARGET'].mean()*100:.1f}%")


# 5. Remove Data Leakage Variables
remove_vars = [
    # Operational/Administrative
    'YEAR', 'CBSAFP_YN', 'FIPSST', 'FORMTYPE', 'FWC', 
    'HHID', 'METRO_YN', 'MPC_YN', 'STRATUM', 'TENURE_IF',
    
    # Target original and follow-up
    'K2Q33A', 'K2Q33B', 'K2Q33C',
    
    # Depression (high comorbidity - bidirectional leakage)
    'K2Q32A', 'K2Q32B', 'K2Q32C',
    
    # Mental health treatment (consequence of diagnosis)
    'K4Q22_R', 'K4Q23', 'K4Q27', 'K4Q28X04',
    
    # CSHCN related (composite includes mental health)
    'SC_CSHCN', 'SC_K2Q13', 'SC_K2Q14', 'SC_K2Q15',
    'SC_K2Q22', 'SC_K2Q23', 'TOTCSHCN', 'TOTNONSHCN',
    
    # Prescription medication related
    'SC_K2Q10', 'SC_K2Q11', 'SC_K2Q12',
    
    # Limited ability (downstream consequence)
    'HCABILITY', 'SC_K2Q16', 'SC_K2Q17', 'SC_K2Q18',
    
    # Other leakage
    'MEMORYCOND', 'C4Q04', 'ARRANGEHC',

    # Survey/Admin
    'HHLANGUAGE', 'BIRTH_MO', 'BIRTH_YR',
    
    # Derived age variables (redundant)
    'TOTAGE_0_5', 'TOTAGE_6_11', 'TOTAGE_12_17',
    'SC_AGE_LT4', 'SC_AGE_LT6', 'SC_AGE_LT9', 'SC_AGE_LT10',
    
    # Imputation flags
    'SC_RACE_R_IF', 'SC_HISPANIC_R_IF', 'SC_SEX_IF',
    'FPL_IF', 'A1_GRADE_IF', 'BIRTH_YR_F', 'HHCOUNT_IF',
]

df = df.drop(columns=[c for c in remove_vars if c in df.columns])
print(f"\nAfter removing leakage variables: {df.shape}")


# 6. Handle Missing Values (Mode Imputation)
missing_cols = df.columns[df.isnull().any()]
print(f"\nColumns with missing values: {len(missing_cols)}")

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"After imputation: {df.shape}")
print(f"Remaining missing: {df.isnull().sum().sum()}")


# 7. Validate No Data Leakage (Cramér's V)
def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = len(x)
    min_dim = min(contingency.shape) - 1
    return 0 if min_dim == 0 else np.sqrt(chi2 / (n * min_dim))

results = {}
for col in df.columns:
    if col != 'TARGET':
        try:
            mask = df[col].notna() & df['TARGET'].notna()
            if mask.sum() > 0:
                results[col] = cramers_v(df.loc[mask, col], df.loc[mask, 'TARGET'])
        except:
            pass

cramers_df = pd.DataFrame({
    'Variable': results.keys(),
    'Cramers_V': results.values()
}).sort_values('Cramers_V', ascending=False)

print("\nTop 30 correlations with TARGET:")
print(cramers_df.head(30).to_string(index=False))

high_corr = cramers_df[cramers_df['Cramers_V'] > 0.5]
print(f"\nVariables with Cramér's V > 0.5: {len(high_corr)}")


# 8. Save Preprocessed Data
df.to_excel("anxiety_preprocessed_FINAL.xlsx", index=False)

print(f"\nFinal shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Anxiety prevalence: {df['TARGET'].mean()*100:.1f}%")
