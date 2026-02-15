# ============================================================
# Student Dropout Prediction - Streamlit App
# ML Assignment 2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout Prediction System")
st.write("Machine Learning Assignment 2 - Model Comparison & Prediction")

# ============================================================
# Upload Dataset
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Student Dropout Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ========================================================
    # Preprocessing
    # ========================================================

    # Target column
    target_column = "Target"

    # Convert target to binary
    df[target_column] = df[target_column].map({
        "Graduate": 1,
        "Dropout": 0,
        "Enrolled": 1   # treating enrolled as non-dropout
    })

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========================================================
    # Models
    # ========================================================

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(),

        "KNN":
        KNeighborsClassifier(),

        "Naive Bayes":
        GaussianNB(),

        "Random Forest":
        RandomForestClassifier(),

        "XGBoost":
        XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # ========================================================
    # Train Models and Evaluate
    # ========================================================

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(y_test, y_pred),

            "AUC": roc_auc_score(y_test, y_prob),

            "Precision": precision_score(y_test, y_pred),

            "Recall": recall_score(y_test, y_pred),

            "F1": f1_score(y_test, y_pred),

            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)

    # ========================================================
    # Show Results Table
    # ========================================================

    st.subheader("Model Comparison Table")

    st.dataframe(
        results_df.sort_values(by="Accuracy", ascending=False),
        use_container_width=True
    )

    # ========================================================
    # Best Model
    # ========================================================

    best_model_name = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]["Model"]

    st.success(f"Best Model: {best_model_name}")

    best_model = models[best_model_name]

    # ========================================================
    # Confusion Matrix
    # ========================================================

    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix")

    st.write(cm)

    # ========================================================
    # Prediction Section
    # ========================================================

    st.subheader("Make Prediction")

    sample_index = st.number_input(
        "Enter sample index",
        min_value=0,
        max_value=len(X_test)-1,
        value=0
    )

    if st.button("Predict"):

        sample = X_test[int(sample_index)].reshape(1,-1)

        prediction = best_model.predict(sample)[0]

        if prediction == 1:
            st.success("Prediction: Student will Graduate / Continue")
        else:
            st.error("Prediction: Student will Dropout")

else:
    st.info("Please upload dataset to continue")