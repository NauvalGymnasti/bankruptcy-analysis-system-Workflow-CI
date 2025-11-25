# modelling.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import argparse


# ---------------------------
# 1. Load Dataset
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data = pd.read_csv(args.data_path)

X = data.drop("Bankrupt?", axis=1)
y = data["Bankrupt?"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 2. MLflow Config
# ---------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SMSML_XGBoost_Tuning")

# ---------------------------
# 3. GridSearch Hyperparameters
# ---------------------------
param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1.0],
}

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("🚀 Starting Grid Search...")
grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1
)

# ---------------------------
# 4. Fit GridSearch (NO MLflow here!)
# ---------------------------
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_
best_score = grid.best_score_

print("\n🎯 Best Parameters:", best_params)
print(f"CV Accuracy: {best_score:.4f}")

# ---------------------------
# 5. Evaluate on Test Set
# ---------------------------
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 6. Start MLflow run AFTER training
# ---------------------------
# Ensure no active run exists
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="XGBoost_GridSearch_Best") as run:

    # Log params & metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_accuracy", best_score)
    mlflow.log_metric("test_accuracy", test_acc)

    # Save artifacts
    os.makedirs("results", exist_ok=True)

    with open("results/classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    with open("results/confusion_matrix.txt", "w") as f:
        f.write(str(confusion_matrix(y_test, y_pred)))

    mlflow.log_artifact("results/classification_report.txt")
    mlflow.log_artifact("results/confusion_matrix.txt")

    # Log MLflow model
    mlflow.xgboost.log_model(
        xgb_model=best_model,
        artifact_path="best_model",
        input_example=X_train.iloc[:1]
    )

    # Save local model
    os.makedirs("model", exist_ok=True)
    best_model.save_model("model/xgb_best_model_tuned.json")

print("\n✅ Model tuned successfully saved & logged to MLflow.")
