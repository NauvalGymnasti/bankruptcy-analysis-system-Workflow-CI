import argparse
import os
import warnings
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(40)


def load_data(path):
    if not os.path.isabs(path):
        # assume path relative to this script
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost classifier via MLproject")
    parser.add_argument("n_estimators", nargs="?", type=int, default=505)
    parser.add_argument("max_depth", nargs="?", type=int, default=35)
    parser.add_argument("dataset", nargs="?", type=str, default="train_pca.csv")

    args = parser.parse_args()
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    dataset_path = args.dataset

    # load data
    df = load_data(dataset_path)

    # determine target column: prefer Credit_Score if exists otherwise last column
    if "Credit_Score" in df.columns:
        target_col = "Credit_Score"
    else:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.head(5)

    # start MLflow run
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", os.path.basename(dataset_path))

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log model (xgboost flavor)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"Training completed — accuracy: {accuracy:.4f}")