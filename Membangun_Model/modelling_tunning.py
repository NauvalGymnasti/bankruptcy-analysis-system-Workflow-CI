# modelling_tuning.py
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from preprocessing import load_and_preprocess_data
import pandas as pd

# --- 1. Load Data ---
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# --- 2. Konfigurasi MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")  
mlflow.set_experiment("SMSML_XGBoost_Tuning")

# --- 3. Grid Search Parameter ---
param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1.0],
}

# --- 4. Model dasar ---
base_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# --- 5. Grid Search dengan Cross-Validation ---
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("🚀 Starting Grid Search...")
grid_search.fit(X_train, y_train)

# --- 6. Ambil hasil terbaik ---
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\n🎯 Best Parameters:")
print(best_params)
print(f"CV Accuracy: {best_score:.4f}")

# --- 7. Evaluasi model terbaik pada test set ---
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 8. Logging ke MLflow ---
with mlflow.start_run(run_name="XGBoost_GridSearchCV_Best"):
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_accuracy", best_score)
    mlflow.log_metric("test_accuracy", test_acc)

    # Simpan artefak evaluasi
    os.makedirs("results", exist_ok=True)
    with open("results/classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    with open("results/confusion_matrix.txt", "w") as f:
        f.write(str(confusion_matrix(y_test, y_pred)))

    mlflow.log_artifact("results/classification_report.txt")
    mlflow.log_artifact("results/confusion_matrix.txt")

    # Logging model ke MLflow (dengan input_example agar tidak ada warning)
    input_example = pd.DataFrame(X_train[:1], columns=X_train.columns)
    mlflow.sklearn.log_model(best_model, name="best_model_tuned", input_example=input_example)

    # Simpan model lokal
    os.makedirs("model", exist_ok=True)
    best_model.save_model("model/xgb_best_model_tuned.json")

print("\n✅ Model tuned telah disimpan dan dilog ke MLflow.")
