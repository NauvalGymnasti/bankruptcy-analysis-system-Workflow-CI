import os
import sys
import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.xgboost


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    return accuracy


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ============================
    # Argumen dari MLproject
    # ============================
    # Format:
    # sys.argv[1] = data_path
    # sys.argv[2] = n_estimators
    # sys.argv[3] = max_depth
    # sys.argv[4] = learning_rate
    # sys.argv[5] = subsample
    # ============================

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data_clean.csv"

    n_estimators = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    learning_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    subsample = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8

    # ============================
    # Load dataset
    # ============================
    data = pd.read_csv(data_path)

    X = data.drop("Bankrupt?", axis=1)
    y = data["Bankrupt?"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================
    # Mulai MLflow Run
    # ============================
    mlflow.xgboost.autolog(disable=True)

    with mlflow.start_run():

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "eval_metric": "logloss",
        }

        # Log parameter manual (sesuai instruksi kriteria 3)
        mlflow.log_params(params)

        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy = eval_metrics(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log model XGBoost
        mlflow.xgboost.log_model(
            booster=model.get_booster(),
            artifact_path="model"
        )

        print("===================================")
        print("XGBoost model trained successfully!")
        print(f"Accuracy: {accuracy}")
        print("===================================")
