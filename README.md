# ❤️ Heart Disease Prediction System

## 📌 Project Overview

This project predicts whether a patient is likely to have heart disease using Machine Learning algorithms.

The application allows users to upload patient data, select a trained model, and obtain predictions through an interactive Streamlit web application.

---

## 📂 Dataset

Dataset: Heart Disease Dataset

Features:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

Target:

- 0 → No Heart Disease
- 1 → Heart Disease

---

## 🧠 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

---

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|--------|----------|-----------|--------|----------|------|------|
| Logistic Regression | 0.8585 | 0.8571 | 0.8298 | 0.8432 | 0.8563 | 0.7147 |
| Decision Tree | 0.7415 | 0.7356 | 0.6809 | 0.7072 | 0.7368 | 0.4775 |
| KNN | 0.8439 | 0.8690 | 0.7766 | 0.8202 | 0.8387 | 0.6864 |
| Naive Bayes | 0.8439 | 0.8298 | 0.8298 | 0.8298 | 0.8428 | 0.6856 |
| Random Forest | 0.8488 | 0.8316 | 0.8404 | 0.8360 | 0.8481 | 0.6957 |

---

## 🏆 Best Model

Based on the evaluation metrics,

**Logistic Regression** achieved the highest overall performance with an accuracy of **85.85%**.

---

## Model Observations

| Model | Observation |
|-------|-------------|
| Logistic Regression | Highest overall accuracy (85.85%) and MCC. Best performing model. |
| Decision Tree | Lowest accuracy; likely overfits the data. |
| KNN | Good precision with competitive performance. |
| Naive Bayes | Simple and fast with balanced results. |
| Random Forest | Strong ensemble model with good recall and stable performance. |
| Overall Winner | Logistic Regression achieved the best overall performance. |

## 📁 Project Structure

```
Heart Disease Prediction/
│
├── data/
├── models/
├── screenshots/
├── src/
├── app.py
├── train.py
├── results.csv
├── test_data.csv
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train models

```
python train.py
```

### Run Streamlit

```
streamlit run app.py
```

---

## 📈 Features

- Data preprocessing
- Five Machine Learning models
- Model comparison
- CSV upload
- Heart disease prediction
- Download prediction results
- Interactive Streamlit interface



