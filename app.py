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


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    layout="wide"
)

st.title("Student Dropout & Academic Success Prediction")


# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Dataset (.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)


    # -----------------------------
    # Target Column
    # -----------------------------
    target_column = "target"

    if target_column not in df.columns:
        st.error("Target column 'target' not found in dataset")
        st.stop()


    # -----------------------------
    # Encode Target
    # -----------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])

    X = df.drop(target_column, axis=1)


    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # -----------------------------
    # Feature Scaling
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # -----------------------------
    # Models (Optimized)
    # -----------------------------
    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=500),

        "Decision Tree":
            DecisionTreeClassifier(max_depth=10),

        "KNN":
            KNeighborsClassifier(n_neighbors=5),

        "Naive Bayes":
            GaussianNB(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0
            )
    }


    # -----------------------------
    # Train Models & Evaluate
    # -----------------------------
    st.subheader("Model Evaluation Metrics")

    results = []

    best_model = None
    best_accuracy = 0
    best_model_name = ""


    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
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
            "Accuracy": round(accuracy, 4),
            "AUC": round(auc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })


        # Track Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name


    results_df = pd.DataFrame(results)

    st.dataframe(results_df, use_container_width=True)

    st.success(f"Best Model: {best_model_name}")


    # -----------------------------
    # Confusion Matrix Heatmap
    # -----------------------------
    st.subheader("Confusion Matrix Heatmap")

    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(6, 4))

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
