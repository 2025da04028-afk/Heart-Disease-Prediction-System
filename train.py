from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.evaluate import evaluate_model
from src.save_results import save_results
from src.save_test_data import save_test_data
from src.save_models import save_model
from src.train_models import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_naive_bayes,
    train_random_forest
    )

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = load_data("data/heart_disease.csv")

print("====================================")
print(" Heart Disease Prediction Project ")
print("====================================")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# -----------------------------
# Step 2: Preprocess Dataset
# -----------------------------
X_train, X_test, y_train, y_test = preprocess_data(df)

print("\nData Preprocessed Successfully!")

print("\nTraining Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

print("\nTraining Target   :", y_train.shape)
print("Testing Target    :", y_test.shape)

# -----------------------------
# Step 3: Train Logistic Regression
# -----------------------------
log_model = train_logistic_regression(X_train, y_train)

save_model(log_model, "logistic_regression.pkl")

print("\nLogistic Regression Model Trained Successfully!")
results = []

log_result = evaluate_model(
    log_model,
    X_test,
    y_test,
    "Logistic Regression"
)

results.append(log_result)

# -----------------------------
# Decision Tree
# -----------------------------
dt_model = train_decision_tree(X_train, y_train)

save_model(dt_model, "decision_tree.pkl")

dt_result = evaluate_model(
    dt_model,
    X_test,
    y_test,
    "Decision Tree"
)

results.append(dt_result)

# -----------------------------
# KNN
# -----------------------------
knn_model = train_knn(X_train, y_train)

save_model(knn_model, "knn.pkl")

knn_result = evaluate_model(
    knn_model,
    X_test,
    y_test,
    "KNN"
)

results.append(knn_result)
# -----------------------------
# Naive Bayes
# -----------------------------
nb_model = train_naive_bayes(X_train, y_train)

save_model(nb_model, "naive_bayes.pkl")

nb_result = evaluate_model(
    nb_model,
    X_test,
    y_test,
    "Naive Bayes"
)

results.append(nb_result)
# -----------------------------
# Random Forest
# -----------------------------
rf_model = train_random_forest(X_train, y_train)

save_model(rf_model, "random_forest.pkl")

rf_result = evaluate_model(
    rf_model,
    X_test,
    y_test,
    "Random Forest"
)

results.append(rf_result)
# -----------------------------
# Save Comparison Results
# -----------------------------

save_results(results)
save_test_data(X_test)