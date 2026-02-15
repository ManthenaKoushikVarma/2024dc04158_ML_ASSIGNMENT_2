import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.title("Student Dropout & Academic Success Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # target column fixed
    target_column = "target"

    if target_column not in df.columns:
        st.error("Target column 'target' not found in dataset")
        st.stop()

    # Encode target
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="mlogloss")
    }

    # Train and evaluate
    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc
        })

    results_df = pd.DataFrame(results)

    st.subheader("Evaluation Metrics")
    st.dataframe(results_df, use_container_width=True)

    # Best model
    best_model_name = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]["Model"]

    st.success(f"Best Model: {best_model_name}")

    best_model = models[best_model_name]

    y_pred_best = best_model.predict(X_test)

    # Confusion matrix
    st.subheader("Confusion Matrix Heatmap")

    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
