import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This application predicts whether a patient is likely to have heart disease
using trained Machine Learning models.

### Available Models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
""")

# ------------------------------------
# Sidebar
# ------------------------------------
st.sidebar.title("About")

st.sidebar.info("""
Machine Learning Assignment 2

Algorithms Used

• Logistic Regression

• Decision Tree

• KNN

• Naive Bayes

• Random Forest
""")

# ------------------------------------
# Select Model
# ------------------------------------
model_option = st.selectbox(
    "Select Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    )
)

# ------------------------------------
# Load Model
# ------------------------------------
model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

model = joblib.load(model_paths[model_option])

# ------------------------------------
# Upload CSV
# ------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(test_data.head())

    if st.button("Predict"):

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(test_data)

        result = test_data.copy()
        result["Prediction"] = predictions

        # -----------------------------
        # Load Actual Labels
        # -----------------------------
        y_true = pd.read_csv("test_labels.csv").squeeze()

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(y_true, predictions)

        st.subheader("📌 Confusion Matrix")

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14
                )

        st.pyplot(fig)

        # -----------------------------
        # Classification Report
        # -----------------------------
        st.subheader("📋 Classification Report")

        report = classification_report(
            y_true,
            predictions,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # -----------------------------
        # Prediction Summary
        # -----------------------------
        st.subheader("Prediction Summary")

        prediction_counts = result["Prediction"].value_counts()

        st.bar_chart(prediction_counts)

        # -----------------------------
        # Prediction Results
        # -----------------------------
        st.subheader("Prediction Results")

        st.dataframe(
            result,
            use_container_width=True
        )

        st.success("Prediction completed successfully!")

        # -----------------------------
        # Download
        # -----------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

# ------------------------------------
# Model Performance Comparison
# ------------------------------------
st.subheader("📊 Model Performance Comparison")

results = pd.read_csv("results.csv")

st.dataframe(results, use_container_width=True)