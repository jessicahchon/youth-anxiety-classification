"""
NSCH 2023 Anxiety Detection - Random Forest
Project: Detecting Adolescent Anxiety Using Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, recall_score, precision_score,
                             f1_score, roc_curve, auc)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


# 1. Load Data
df = pd.read_excel("anxiety_preprocessed_FINAL.xlsx")
print(f"Data shape: {df.shape}")

X = df.drop(columns=['TARGET'])
y = df['TARGET']


# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# 3. Evaluation Function
def evaluate_rf(X_tr, y_tr, X_te, y_te, name):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    
    result = {
        'Method': name,
        'Balanced_Acc': balanced_accuracy_score(y_te, y_pred),
        'ROC_AUC': roc_auc_score(y_te, y_prob),
        'Sensitivity': recall_score(y_te, y_pred),
        'Specificity': recall_score(y_te, y_pred, pos_label=0)
    }
    print(f"\n{name}:")
    print(f"  Balanced Acc: {result['Balanced_Acc']:.4f}")
    print(f"  ROC-AUC: {result['ROC_AUC']:.4f}")
    return result


# 4. Compare Sampling Methods
results = []

# Method 1: No Sampling
results.append(evaluate_rf(X_train, y_train, X_test, y_test, "No Sampling"))

# Method 2: SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
results.append(evaluate_rf(X_train_smote, y_train_smote, X_test, y_test, "SMOTE"))

# Method 3: Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
results.append(evaluate_rf(X_train_rus, y_train_rus, X_test, y_test, "Undersampling"))

# Method 4: Class Weight
rf_weighted = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
rf_weighted.fit(X_train, y_train)
y_pred_w = rf_weighted.predict(X_test)
y_prob_w = rf_weighted.predict_proba(X_test)[:, 1]
results.append({
    'Method': 'Class Weight',
    'Balanced_Acc': balanced_accuracy_score(y_test, y_pred_w),
    'ROC_AUC': roc_auc_score(y_test, y_prob_w),
    'Sensitivity': recall_score(y_test, y_pred_w),
    'Specificity': recall_score(y_test, y_pred_w, pos_label=0)
})

# Summary
print("\n" + "="*60)
print("SAMPLING COMPARISON")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_method = results_df.loc[results_df['Balanced_Acc'].idxmax(), 'Method']
print(f"\nBest method: {best_method}")


# 5. Hyperparameter Tuning (with Undersampling)
print("\n" + "="*60)
print("Hyperparameter Tuning")
print("="*60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, n_iter=50, cv=cv, scoring='balanced_accuracy',
    n_jobs=-1, verbose=1, random_state=42
)
random_search.fit(X_train_rus, y_train_rus)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV Balanced Accuracy: {random_search.best_score_:.4f}")

best_rf = random_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print(f"\nTest Set Performance:")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")


# 6. Feature Selection - Find Optimal k
print("\n" + "="*60)
print("Feature Selection - Find Optimal k")
print("="*60)

k_results = []
for k in [50, 100, 150, 180, 200, 220, 239]:
    selector = SelectKBest(f_classif, k=k)
    X_train_k = selector.fit_transform(X_train_rus, y_train_rus)
    X_test_k = selector.transform(X_test)
    
    rf_k = RandomForestClassifier(**random_search.best_params_, random_state=42, n_jobs=-1)
    rf_k.fit(X_train_k, y_train_rus)
    y_pred_k = rf_k.predict(X_test_k)
    y_prob_k = rf_k.predict_proba(X_test_k)[:, 1]
    
    k_results.append({
        'k': k,
        'Balanced_Acc': balanced_accuracy_score(y_test, y_pred_k),
        'ROC_AUC': roc_auc_score(y_test, y_prob_k)
    })

k_df = pd.DataFrame(k_results)
print(k_df.to_string(index=False))
best_k = int(k_df.loc[k_df['Balanced_Acc'].idxmax(), 'k'])
print(f"\nOptimal k: {best_k}")


# 7. Final Model with Optimal k
print("\n" + "="*60)
print(f"Final Model with k={best_k}")
print("="*60)

selector_rf = SelectKBest(f_classif, k=best_k)
X_train_selected = selector_rf.fit_transform(X_train_rus, y_train_rus)
X_test_selected = selector_rf.transform(X_test)
selected_features = X_train.columns[selector_rf.get_support()].tolist()

print(f"Selected features: {len(selected_features)}")

rf_final = RandomForestClassifier(**random_search.best_params_, random_state=42, n_jobs=-1)
rf_final.fit(X_train_selected, y_train_rus)

y_pred_final = rf_final.predict(X_test_selected)
y_prob_final = rf_final.predict_proba(X_test_selected)[:, 1]

print(f"\nBalanced Accuracy: {balanced_accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_final):.4f}")
print(f"Sensitivity: {recall_score(y_test, y_pred_final):.4f}")
print(f"Specificity: {recall_score(y_test, y_pred_final, pos_label=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_final)}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Anxiety', 'Anxiety']))


# 8. Cross-Validation
print("\n" + "="*60)
print("5-Fold Cross-Validation")
print("="*60)

X_selected_full = X[selected_features]
cv_pipeline = ImbPipeline([
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('classifier', RandomForestClassifier(**random_search.best_params_, random_state=42, n_jobs=-1))
])

cv_results = cross_validate(
    cv_pipeline, X_selected_full, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['balanced_accuracy', 'roc_auc'],
    return_train_score=True
)

print(f"Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")
print(f"ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")


# 9. Feature Importance (Top 30)
print("\n" + "="*60)
print("Top 30 Important Features")
print("="*60)

importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_final.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.head(30).to_string(index=False))


# 10. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_rf.png', dpi=300)
plt.show()


# 11. Save Model
joblib.dump(rf_final, 'random_forest_final.pkl')
pd.DataFrame({'Feature': selected_features}).to_csv('selected_features_rf.csv', index=False)
print("\nModel saved: random_forest_final.pkl")
print("Features saved: selected_features_rf.csv")
