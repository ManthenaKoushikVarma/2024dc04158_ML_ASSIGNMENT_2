import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Student Dropout Prediction",
    layout="wide"
)

st.title("🎓 Student Dropout Prediction - ML Assignment")
st.write("BITS Pilani WILP - Machine Learning Assignment 2")

# -------------------------
# LOAD DATA
# -------------------------

try:
    df = pd.read_csv("students_dropout.csv")
except:
    st.error("Dataset file not found. Please ensure CSV is in repo.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# TARGET COLUMN
# -------------------------

target_column = "target"

if target_column not in df.columns:
    st.error("Target column 'target' not found in dataset.")
    st.stop()

# -------------------------
# FEATURE & TARGET SPLIT
# -------------------------

X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# SCALING
# -------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# MODELS
# -------------------------

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
        XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss'
        )
}

# -------------------------
# TRAIN & EVALUATE
# -------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({

        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# -------------------------
# SHOW METRICS TABLE
# -------------------------

st.subheader("Model Evaluation Metrics")

st.dataframe(
    results_df.sort_values("Accuracy", ascending=False),
    use_container_width=True
)

# -------------------------
# BEST MODEL
# -------------------------

best_model_name = results_df.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]

st.success(f"Best Model: {best_model_name}")

# -------------------------
# CONFUSION MATRIX
# -------------------------

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_model_name}")

st.pyplot(fig)

# -------------------------
# SHOW METRICS FOR BEST MODEL
# -------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Best Model Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")

col1.metric("F1 Score", f"{f1:.4f}")
col2.metric("MCC", f"{mcc:.4f}")