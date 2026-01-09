"""
NSCH 2023 Anxiety Detection - External Validation (2024 Data)
Project: Detecting Adolescent Anxiety Using Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, recall_score, precision_score,
                             f1_score, roc_curve, auc)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


# 1. Load Saved Models
print("="*60)
print("Load Saved Models")
print("="*60)

lr_model = joblib.load('logistic_regression_final.pkl')
rf_model = joblib.load('random_forest_final.pkl')
xgb_model = joblib.load('xgboost_final.pkl')

print("Loaded: logistic_regression_final.pkl")
print("Loaded: random_forest_final.pkl")
print("Loaded: xgboost_final.pkl")

# Load selected features for each model
features_lr = pd.read_csv('selected_features_logistic.csv')['Feature'].tolist()
features_rf = pd.read_csv('selected_features_rf.csv')['Feature'].tolist()
features_xgb = pd.read_csv('selected_features_xgb.csv')['Feature'].tolist()

print(f"\nLR features: {len(features_lr)}")
print(f"RF features: {len(features_rf)}")
print(f"XGB features: {len(features_xgb)}")


# 2. Load and Preprocess 2024 Validation Data
print("\n" + "="*60)
print("Load and Preprocess Validation Data (2024)")
print("="*60)

df_2024 = pd.read_excel("nsch_2024_topical.xlsx")
print(f"2024 Original: {df_2024.shape}")

# Filter age 12-17
df_2024 = df_2024[(df_2024['SC_AGE_YEARS'] >= 12) & (df_2024['SC_AGE_YEARS'] <= 17)].copy()
print(f"After age filter (12-17): {df_2024.shape}")

# Create TARGET
df_2024 = df_2024[df_2024['K2Q33A'].isin([1, 2])].copy()
df_2024['TARGET'] = (df_2024['K2Q33A'] == 1).astype(int)
print(f"After TARGET creation: {df_2024.shape}")
print(f"2024 Prevalence: {df_2024['TARGET'].mean()*100:.1f}%")

y_val_2024 = df_2024['TARGET']


# 3. Check Feature Availability
print("\n" + "="*60)
print("Feature Availability Check")
print("="*60)

missing_lr = [f for f in features_lr if f not in df_2024.columns]
missing_rf = [f for f in features_rf if f not in df_2024.columns]
missing_xgb = [f for f in features_xgb if f not in df_2024.columns]

print(f"LR - Missing features: {len(missing_lr)} {missing_lr if missing_lr else ''}")
print(f"RF - Missing features: {len(missing_rf)} {missing_rf if missing_rf else ''}")
print(f"XGB - Missing features: {len(missing_xgb)} {missing_xgb if missing_xgb else ''}")

# Use common features only
common_lr = [f for f in features_lr if f in df_2024.columns]
common_rf = [f for f in features_rf if f in df_2024.columns]
common_xgb = [f for f in features_xgb if f in df_2024.columns]


# 4. Prepare Validation Sets
print("\n" + "="*60)
print("Prepare Validation Sets")
print("="*60)

def prepare_validation_data(df, features):
    X = df[features].copy()
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])
    return X

X_val_lr = prepare_validation_data(df_2024, common_lr)
X_val_rf = prepare_validation_data(df_2024, common_rf)
X_val_xgb = prepare_validation_data(df_2024, common_xgb)

print(f"LR validation set: {X_val_lr.shape}")
print(f"RF validation set: {X_val_rf.shape}")
print(f"XGB validation set: {X_val_xgb.shape}")


# 5. External Validation - All Models
print("\n" + "="*60)
print("External Validation Results")
print("="*60)

results = []

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
y_pred_lr = lr_model.predict(X_val_lr)
y_prob_lr = lr_model.predict_proba(X_val_lr)[:, 1]

print(f"Balanced Accuracy: {balanced_accuracy_score(y_val_2024, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val_2024, y_prob_lr):.4f}")
print(f"Sensitivity: {recall_score(y_val_2024, y_pred_lr):.4f}")
print(f"Specificity: {recall_score(y_val_2024, y_pred_lr, pos_label=0):.4f}")
print(f"Precision: {precision_score(y_val_2024, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_val_2024, y_pred_lr):.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Balanced_Acc': balanced_accuracy_score(y_val_2024, y_pred_lr),
    'ROC_AUC': roc_auc_score(y_val_2024, y_prob_lr),
    'Sensitivity': recall_score(y_val_2024, y_pred_lr),
    'Specificity': recall_score(y_val_2024, y_pred_lr, pos_label=0),
    'Precision': precision_score(y_val_2024, y_pred_lr),
    'F1': f1_score(y_val_2024, y_pred_lr)
})


# --- Random Forest ---
print("\n--- Random Forest ---")
y_pred_rf = rf_model.predict(X_val_rf)
y_prob_rf = rf_model.predict_proba(X_val_rf)[:, 1]

print(f"Balanced Accuracy: {balanced_accuracy_score(y_val_2024, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val_2024, y_prob_rf):.4f}")
print(f"Sensitivity: {recall_score(y_val_2024, y_pred_rf):.4f}")
print(f"Specificity: {recall_score(y_val_2024, y_pred_rf, pos_label=0):.4f}")
print(f"Precision: {precision_score(y_val_2024, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_val_2024, y_pred_rf):.4f}")

results.append({
    'Model': 'Random Forest',
    'Balanced_Acc': balanced_accuracy_score(y_val_2024, y_pred_rf),
    'ROC_AUC': roc_auc_score(y_val_2024, y_prob_rf),
    'Sensitivity': recall_score(y_val_2024, y_pred_rf),
    'Specificity': recall_score(y_val_2024, y_pred_rf, pos_label=0),
    'Precision': precision_score(y_val_2024, y_pred_rf),
    'F1': f1_score(y_val_2024, y_pred_rf)
})


# --- XGBoost ---
print("\n--- XGBoost ---")
y_pred_xgb = xgb_model.predict(X_val_xgb)
y_prob_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]

print(f"Balanced Accuracy: {balanced_accuracy_score(y_val_2024, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val_2024, y_prob_xgb):.4f}")
print(f"Sensitivity: {recall_score(y_val_2024, y_pred_xgb):.4f}")
print(f"Specificity: {recall_score(y_val_2024, y_pred_xgb, pos_label=0):.4f}")
print(f"Precision: {precision_score(y_val_2024, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_val_2024, y_pred_xgb):.4f}")

results.append({
    'Model': 'XGBoost',
    'Balanced_Acc': balanced_accuracy_score(y_val_2024, y_pred_xgb),
    'ROC_AUC': roc_auc_score(y_val_2024, y_prob_xgb),
    'Sensitivity': recall_score(y_val_2024, y_pred_xgb),
    'Specificity': recall_score(y_val_2024, y_pred_xgb, pos_label=0),
    'Precision': precision_score(y_val_2024, y_pred_xgb),
    'F1': f1_score(y_val_2024, y_pred_xgb)
})


# 6. Summary Comparison
print("\n" + "="*60)
print("EXTERNAL VALIDATION SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Performance comparison (2023 Test → 2024 Validation)
print("\n--- Performance Change (2023 Test → 2024 Validation) ---")
print(f"LR:  0.789 → {results[0]['Balanced_Acc']:.3f} ({results[0]['Balanced_Acc']-0.789:+.3f})")
print(f"RF:  0.789 → {results[1]['Balanced_Acc']:.3f} ({results[1]['Balanced_Acc']-0.789:+.3f})")
print(f"XGB: 0.793 → {results[2]['Balanced_Acc']:.3f} ({results[2]['Balanced_Acc']-0.793:+.3f})")


# 7. ROC Curves - All Models
print("\n" + "="*60)
print("ROC Curves")
print("="*60)

fpr_lr, tpr_lr, _ = roc_curve(y_val_2024, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_val_2024, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val_2024, y_prob_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
         label=f'Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, 
         label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.3f})')
plt.plot(fpr_xgb, tpr_xgb, color='red', lw=2, 
         label=f'XGBoost (AUC = {auc(fpr_xgb, tpr_xgb):.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves - External Validation (2024 Data)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_external_validation.png', dpi=300)
plt.show()

print("\nSaved: roc_curve_external_validation.png")


# 8. Save Results
results_df.to_csv('external_validation_results.csv', index=False)
print("Results saved: external_validation_results.csv")
