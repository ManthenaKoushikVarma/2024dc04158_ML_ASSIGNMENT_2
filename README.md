# Student Dropout Prediction – Machine Learning Assignment

## Problem Statement

Student dropout is a major concern for educational institutions as it impacts academic performance, institutional reputation, and student success. Early identification of students who are at risk of dropping out can help institutions take preventive actions.

The goal of this project is to build and compare multiple Machine Learning models to predict whether a student will **Dropout (0)** or **Graduate (1)** based on academic, demographic, and financial features.

This is a **binary classification problem**.

---

## Dataset Description

The dataset used is the **Student Dropout and Academic Success Dataset**.

**Dataset characteristics:**

- Number of records: 4424 students
- Number of features: 36 input features + 1 target variable
- Target variable: `Target`
  - Dropout = 0
  - Graduate = 1

**Feature categories include:**

- Demographic information
- Academic performance
- Financial information
- Enrollment details
- Previous qualifications

The dataset was preprocessed and converted into a binary classification problem by removing students with "Enrolled" status.

---

## Models Used and Performance Comparison

The following Machine Learning models were trained and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Model Comparison Table

| ML Model Name       | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------- | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.910468 | 0.953583 | 0.906780  | 0.953229 | 0.929425 | 0.809107 |
| Decision Tree       | 0.841598 | 0.832524 | 0.872768  | 0.870824 | 0.871795 | 0.664591 |
| KNN                 | 0.858127 | 0.896244 | 0.833977  | 0.962138 | 0.893485 | 0.700154 |
| Naive Bayes         | 0.849862 | 0.879898 | 0.851240  | 0.917595 | 0.883173 | 0.677702 |
| Random Forest       | 0.898072 | 0.953185 | 0.891441  | 0.951002 | 0.920259 | 0.782580 |
| XGBoost             | 0.900826 | 0.958850 | 0.905376  | 0.937639 | 0.921225 | 0.788386 |

---

## Observations on Model Performance

### Logistic Regression

Logistic Regression achieved the highest accuracy of **91.05%** and strong performance across all metrics. It demonstrates that the dataset has strong linear relationships and is well suited for linear classification.

### Decision Tree

Decision Tree achieved moderate performance with **84.16% accuracy**. It is easy to interpret but prone to overfitting, which may limit generalization performance.

### K-Nearest Neighbors (KNN)

KNN achieved **85.81% accuracy** and very high recall (**96.21%**), meaning it is effective at identifying dropout students but slightly lower precision suggests more false positives.

### Naive Bayes

Naive Bayes achieved **84.99% accuracy** and performed reasonably well. Its assumption of feature independence limits its ability to fully capture feature relationships.

### Random Forest (Ensemble)

Random Forest achieved strong performance with **89.81% accuracy** and high AUC. As an ensemble model, it reduces overfitting and improves prediction stability.

### XGBoost (Ensemble)

XGBoost achieved **90.08% accuracy** and the highest AUC (**0.958850**), making it one of the best performing models. It effectively captures complex feature interactions and provides strong predictive performance.

---

## Best Model

Based on overall performance and accuracy:

**Best Model: Logistic Regression**

Accuracy: **91.05%**  
AUC: **0.953583**  
MCC: **0.809107**

Logistic Regression performed best overall while maintaining simplicity and strong generalization ability.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Streamlit
- Jupyter Notebook

---

## Streamlit Deployment

The trained model is deployed using Streamlit to allow interactive prediction and visualization.

Users can upload the dataset and view model predictions and performance.
