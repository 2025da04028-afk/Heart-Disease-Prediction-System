import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
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

st.sidebar.title("About")

st.sidebar.info("""
**Machine Learning Assignment 2**

Heart Disease Prediction

Algorithms Used:
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
""")

st.write("Upload the test dataset and select a trained model.")

# -----------------------------
# Model Selection
# -----------------------------
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

# -----------------------------
# Load Selected Model
# -----------------------------
model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

model = joblib.load(model_paths[model_option])

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")

    st.dataframe(test_data.head())

    if st.button("Predict"):

        predictions = model.predict(test_data)

        result = test_data.copy()

        result["Prediction"] = predictions
        st.subheader("Prediction Summary")

        prediction_counts = result["Prediction"].value_counts()

        st.bar_chart(prediction_counts)

        st.subheader("Prediction Results")

        st.subheader("Prediction Results")

        st.dataframe(result, use_container_width=True)

        st.success("Prediction completed successfully!")

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

# -----------------------------
# Display Model Performance
# -----------------------------
results = pd.read_csv("results.csv")

st.subheader("📊 Model Performance")

selected_result = results[results["Model"] == model_option]

st.dataframe(selected_result, use_container_width=True)