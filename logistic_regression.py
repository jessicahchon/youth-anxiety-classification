"""
NSCH 2023 Anxiety Detection - Logistic Regression
Project: Detecting Adolescent Anxiety Using Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
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
print(f"TARGET distribution:\n{df['TARGET'].value_counts()}")
print(f"Prevalence: {df['TARGET'].mean()*100:.1f}%")

X = df.drop(columns=['TARGET'])
y = df['TARGET']


# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# 3. Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*60}")
    print(f"Results: {model_name}")
    print(f"{'='*60}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Sensitivity (Recall): {recall_score(y_test, y_pred):.4f}")
    print(f"Specificity: {recall_score(y_test, y_pred, pos_label=0):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return {
        'model': model_name,
        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'sensitivity': recall_score(y_test, y_pred),
        'specificity': recall_score(y_test, y_pred, pos_label=0),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


# 4. Compare Sampling Methods
results = []

# Method 1: No sampling (baseline)
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train, y_train)
results.append(evaluate_model(lr_baseline, X_test, y_test, "No Sampling"))

# Method 2: SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
results.append(evaluate_model(lr_smote, X_test, y_test, "SMOTE"))

# Method 3: Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
lr_rus = LogisticRegression(max_iter=1000, random_state=42)
lr_rus.fit(X_train_rus, y_train_rus)
results.append(evaluate_model(lr_rus, X_test, y_test, "Undersampling"))

# Method 4: Class Weight
lr_weighted = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_weighted.fit(X_train, y_train)
results.append(evaluate_model(lr_weighted, X_test, y_test, "Class Weight"))

# Comparison Summary
print("\n" + "="*60)
print("SAMPLING COMPARISON")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = results_df.loc[results_df['balanced_acc'].idxmax()]
print(f"\nBest method: {best['model']} (Balanced Acc: {best['balanced_acc']:.4f})")


# 5. Hyperparameter Tuning (with Undersampling)
print("\n" + "="*60)
print("Hyperparameter Tuning")
print("="*60)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_rus, y_train_rus)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV Balanced Accuracy: {grid_search.best_score_:.4f}")

best_lr = grid_search.best_estimator_
evaluate_model(best_lr, X_test, y_test, "Tuned Logistic Regression")


# 6. Feature Selection
print("\n" + "="*60)
print("Feature Selection - Find Optimal k")
print("="*60)

k_results = []
for k in [100, 150, 180, 200, 220, 239]:
    selector = SelectKBest(f_classif, k=k)
    X_train_k = selector.fit_transform(X_train_rus, y_train_rus)
    X_test_k = selector.transform(X_test)
    
    lr_k = LogisticRegression(**grid_search.best_params_, max_iter=2000, random_state=42)
    lr_k.fit(X_train_k, y_train_rus)
    y_pred_k = lr_k.predict(X_test_k)
    y_prob_k = lr_k.predict_proba(X_test_k)[:, 1]
    
    k_results.append({
        'k': k,
        'Balanced_Acc': balanced_accuracy_score(y_test, y_pred_k),
        'ROC_AUC': roc_auc_score(y_test, y_prob_k)
    })

k_df = pd.DataFrame(k_results)
print(k_df.to_string(index=False))
best_k = int(k_df.loc[k_df['Balanced_Acc'].idxmax(), 'k'])
print(f"\nOptimal k: {best_k}")

# Apply SelectKBest with optimal k
selector_kbest = SelectKBest(f_classif, k=best_k)
X_train_kbest = selector_kbest.fit_transform(X_train_rus, y_train_rus)
X_test_kbest = selector_kbest.transform(X_test)
selected_features = X_train.columns[selector_kbest.get_support()].tolist()


# 7. Final Model Evaluation
print("\n" + "="*60)
print("Final Model Evaluation")
print("="*60)

final_model = LogisticRegression(**grid_search.best_params_, max_iter=2000, random_state=42)
final_model.fit(X_train_kbest, y_train_rus)

y_pred_final = final_model.predict(X_test_kbest)
y_prob_final = final_model.predict_proba(X_test_kbest)[:, 1]

print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_final):.4f}")
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

X_selected = X[selected_features]
cv_pipeline = ImbPipeline([
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('classifier', LogisticRegression(**grid_search.best_params_, max_iter=2000, random_state=42))
])

cv_results = cross_validate(
    cv_pipeline, X_selected, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['balanced_accuracy', 'roc_auc'],
    return_train_score=True
)

print(f"Balanced Accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")
print(f"ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")


# 9. Feature Importance (Top 20)
print("\n" + "="*60)
print("Top 20 Important Features")
print("="*60)

coef_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': final_model.coef_[0],
    'Abs_Coefficient': np.abs(final_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(coef_df.head(20).to_string(index=False))


# 10. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_logistic.png', dpi=300)
plt.show()


# 11. Save Model
joblib.dump(final_model, 'logistic_regression_final.pkl')
pd.DataFrame({'Feature': selected_features}).to_csv('selected_features_logistic.csv', index=False)
print("\nModel saved: logistic_regression_final.pkl")
print("Features saved: selected_features_logistic.csv")
