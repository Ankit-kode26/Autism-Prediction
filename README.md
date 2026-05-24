# 🧩 Autism Prediction Using Machine Learning

A complete end-to-end machine learning pipeline to predict Autism Spectrum Disorder (ASD) traits based on behavioral, demographic, and screening data. The goal is to build an effective predictive model that handles a highly skewed dataset to correctly identify clinical cases.

## 📁 Project Workflow

- **Data Collection & Inspection** — Sourced tabular screening data from Kaggle
- **Exploratory Data Analysis (EDA)** — Visualized distributions and discovered class skewness
- **Data Preprocessing** — Cleaned missing data, encoded categorical variables, handled class imbalance using SMOTE
- **Model Selection & Tuning** — Trained multiple ensemble algorithms with `RandomizedSearchCV`
- **Evaluation** — Benchmarked using cross-validation, precision, and recall metrics

## 📊 Dataset

1.  Source : Kaggle Autism Prediction Dataset
2. Target Variable : `class_ASD` — `0` = No ASD, `1` = ASD
3. Key Features : Questionnaire responses, ethnicity, family history, jaundice indicators
4. Challenge : Severe class imbalance (non-ASD >> ASD instances)

Note:  `age_desc` was dropped as it contained only one unique value across all records.

## 🛠️ Tech Stack

1.Environment : VS Code / Jupyter Notebook
2. Language : Python
3. Data Handling : Pandas, NumPy
4. Visualization : Matplotlib, Seaborn
5. ML & Preprocessing : scikit-learn, imbalanced-learn, XGBoost

## ⚙️ Key ML Operations

**🔁 Handling Class Imbalance**
Used `SMOTE` (Synthetic Minority Over-sampling Technique) from `imbalanced-learn` to synthetically generate minority class samples, preventing the model from being biased toward the majority class.

**🔤 Feature Encoding**
Applied `LabelEncoder` to transform categorical text features (gender, country of residence, etc.) into structured numerical values.

**🎯 Hyperparameter Tuning**
Used `RandomizedSearchCV` instead of Grid Search to efficiently explore a large parameter space while saving significant computation time.

## 🙏 Acknowledgements

1. Dataset: [Kaggle Autism Screening Dataset](https://www.kaggle.com/)
2. Original project walkthrough: [YouTube Tutorial](https://youtu.be/xwaUuAR-W1A)
