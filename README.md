# 📉 Telco Customer Churn Prediction

## 📝 Description

This machine learning project predicts whether a customer is likely to churn (cancel a service) using demographic, service usage, and account-related features from the **Telco Customer Churn** dataset. A Random Forest Classifier with hyperparameter tuning is used for classification.

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

## ✅ Features

- Data preprocessing and cleaning:
  - Handles missing values in `TotalCharges`
  - Label encodes categorical features
- Scales numerical features using `StandardScaler`
- Splits dataset into training and testing sets
- Trains a `RandomForestClassifier` with `GridSearchCV` to optimize:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
- Evaluates model using:
  - Accuracy
  - Classification report (Precision, Recall, F1-score)
  - Confusion matrix visualization

## 📊 Dataset

- 📁 File: `Telco-Customer-Churn.csv`  
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Target:
- `Churn` (Yes/No → 1/0)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
