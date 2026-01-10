# NSCH 2023 Adolescent Anxiety Detection Using Machine Learning

## Overview
This project develops machine learning models to detect anxiety disorders in U.S. adolescents (ages 12-17) using the National Survey of Children's Health (NSCH) 2023 data. The models are externally validated using NSCH 2024 data to assess temporal generalizability.

## Authors
- **Jessica Chon** (Research Lead) - M.S. Applied Statistics, California State University, Long Beach
- Hojun Yim - Bishop Montgomery High School
- Esther Yi  
- Rachel Kwak - Van Nuys High School
- Sydney An - Marlborough School

## Dataset
- **Training Data**: NSCH 2023 (n = 18,283 adolescents aged 12-17)
- **External Validation**: NSCH 2024 (n = 21,821 adolescents aged 12-17)
- **Target Variable**: Parent-reported anxiety diagnosis (K2Q33A)
- **Prevalence**: 23.3% in both datasets

Data source: [NSCH Data Resource Center](https://www.childhealthdata.org/)

## Methodology

### Data Preprocessing
1. Filter adolescents aged 12-17
2. Remove variables with >30% missing values
3. Remove 55 data leakage variables (depression diagnosis, mental health treatment, CSHCN screener, etc.)
4. Mode imputation for remaining missing values
5. Final dataset: 239 features

### Class Imbalance Handling
Compared four methods:
- No sampling (baseline)
- SMOTE
- Random Undersampling ✓ (selected)
- Class weights

### Machine Learning Models
| Model | Hyperparameter Tuning | Feature Selection |
|-------|----------------------|-------------------|
| Logistic Regression | GridSearchCV (C, penalty, solver) | SelectKBest (k=220) |
| Random Forest | RandomizedSearchCV (n_estimators, max_depth, etc.) | SelectKBest (k=100) |
| XGBoost | RandomizedSearchCV (learning_rate, max_depth, etc.) | SelectKBest (k=180) |

### Evaluation Metrics
- Primary: Balanced Accuracy, ROC-AUC
- Secondary: Sensitivity, Specificity, Precision, F1-Score
- Validation: 5-Fold Stratified Cross-Validation

## Results

### Internal Validation (2023 Test Set)
| Model | Balanced Accuracy | ROC-AUC | Sensitivity | Specificity |
|-------|------------------|---------|-------------|-------------|
| Logistic Regression | 0.789 | 0.876 | 0.763 | 0.815 |
| Random Forest | 0.789 | 0.867 | 0.802 | 0.776 |
| XGBoost | 0.793 | 0.876 | 0.778 | 0.808 |

### External Validation (2024 Data)
| Model | Balanced Accuracy | ROC-AUC | Sensitivity | Specificity |
|-------|------------------|---------|-------------|-------------|
| Logistic Regression | 0.793 | 0.876 | 0.771 | 0.816 |
| Random Forest | 0.784 | 0.867 | 0.789 | 0.780 |
| XGBoost | 0.793 | 0.876 | 0.777 | 0.810 |

All models demonstrated stable performance on external validation, indicating good temporal generalizability.

### Top Predictive Features
1. MAKEFRIEND - Difficulty making friends
2. VIDEOPHONE - Video/phone communication frequency
3. K7Q85_R - Argues too much
4. K2Q31A - ADD/ADHD diagnosis
5. DECISIONS_R - Difficulty with decisions
6. SC_SEX - Sex of child
7. HEADACHE - Frequent headaches
8. BULLIED_R - Bullying experience
9. ENGAGE_INTEREST - Interest/curiosity engagement
10. ACE8 - Adverse childhood experience (violence exposure)

## File Structure
```
├── data_preprocessing.py          # Data cleaning and preprocessing
├── logistic_regression.py         # Logistic Regression model
├── random_forest.py               # Random Forest model
├── xgboost_model.py               # XGBoost model
├── external_validation.py         # External validation with 2024 data
├── logistic_regression_final.pkl  # Saved LR model
├── random_forest_final.pkl        # Saved RF model
├── xgboost_final.pkl              # Saved XGBoost model
├── selected_features_logistic.csv # Selected features for LR
├── selected_features_rf.csv       # Selected features for RF
├── selected_features_xgb.csv      # Selected features for XGBoost
├── anxiety_preprocessed_FINAL.xlsx # Preprocessed 2023 data
├── external_validation_results.csv # External validation results
├── roc_curve_logistic.png         # ROC curve - LR
├── roc_curve_rf.png               # ROC curve - RF
├── roc_curve_xgb.png              # ROC curve - XGBoost
├── roc_curve_external_validation.png # ROC curves - External validation
└── README.md
```

## Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
joblib>=1.2.0
openpyxl>=3.0.0
```

## Usage

### 1. Data Preprocessing
```bash
python data_preprocessing.py
```

### 2. Train Models
```bash
python logistic_regression.py
python random_forest.py
python xgboost_model.py
```

### 3. External Validation
```bash
python external_validation.py
```

## Citation
```
Chon, J., Yim, H., Yi, E., Kwak, R., & An, S. (2026). Machine Learning Approaches 
for Detecting Adolescent Anxiety Using the National Survey of Children's Health. 
California State University, Long Beach.
```

## Acknowledgments
Data provided by the National Survey of Children's Health (NSCH), U.S. Census Bureau and Health Resources and Services Administration.
Code documentation assisted by Claude AI.

## License
This project is for academic research purposes.
