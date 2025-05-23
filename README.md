# 💳 Credit Card Fraud Detection: A CRISP-DM Approach

A comprehensive machine learning pipeline to detect fraudulent credit card transactions using real-world anonymized data. This project uses the CRISP-DM methodology to explore, clean, model, and evaluate solutions to a classic class-imbalance problem.

---

## 📌 Project Overview

Credit card fraud detection is a critical challenge faced by financial institutions. This project tackles it as an **anomaly detection problem**, leveraging both **SMOTE** for class balancing and a variety of machine learning models to find the best fit. The final solution aims for high recall on fraud cases while minimizing false positives.

---

## 🧠 Objectives

- Identify **when** fraud typically happens (e.g., time of day).
- Determine **patterns in transaction amounts** for fraudulent vs. legitimate transactions.
- Implement **data balancing** using SMOTE to handle extreme class imbalance.
- Compare and evaluate **multiple ML classifiers** using precision, recall, AUC-ROC, and F1-score.

---

## 📂 Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions  
- **Features**: 30 numerical features (V1–V28, Time, Amount), 1 binary target (`Class`)  
  - `Class = 0`: Legitimate  
  - `Class = 1`: Fraud

---

## 🧹 Data Preparation

- No missing values.
- All features are numeric; only `Class` is categorical.
- Data standardized using `StandardScaler`.
- SMOTE used to balance the dataset before modeling.

---

## 📊 Exploratory Insights

- **99.8% of transactions are under $2500**.
- **All fraudulent transactions are below $2500**, indicating an intentional tactic to avoid detection.
- No significant temporal pattern observed for fraud occurrence.

---

## 🛠️ Modeling Pipeline

- Train-Test Split: 70:30
- Feature Selection: Mutual Information Score
- Class Balancing: SMOTE
- Models Used:
  - Logistic Regression
  - SGD Classifier
  - K-Nearest Neighbors
  - Random Forest
- GridSearchCV used for hyperparameter tuning
- Pipelines created using `sklearn.pipeline.Pipeline`

---

## 🧪 Evaluation Metrics

Each model was evaluated using:

- **F1 Score**
- **Accuracy**
- **AUC-ROC**
- **Classification Report**

Best-performing model:
- **K-Nearest Neighbors (p=2)**  
  - Accuracy: **99.82%**  
  - F1-Score: **0.63** (Fraud class)  
  - Strong performance with balanced data

---

## 🧾 Key Findings

- Fraud detection is most accurate when using SMOTE-balanced data.
- Fraudulent transactions consistently involve **lower amounts**.
- **KNN** with Euclidean distance performed the best, followed by **Random Forest**.
- Time is not a strong predictor of fraud.

---

## 📁 Directory Structure

