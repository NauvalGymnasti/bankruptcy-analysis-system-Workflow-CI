import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

def run_model(data_path: str, target_column: str):
    # =====================================================================
    # Load dataset (SUDAH BERSIH)
    # =====================================================================
    df = pd.read_csv(data_path)
    target_column= "Bankrupt?"

    # Pastikan kolom target ada
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")

    # =====================================================================
    # Pisahkan fitur vs target TANPA drop kolom yang tidak ada
    # =====================================================================
    X = df.drop(columns=[target_column], errors="ignore")  
    y = df[target_column]

    # Convert all non-numeric → numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # =====================================================================
    # Train-test split
    # =====================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =====================================================================
    # Train model
    # =====================================================================
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # =====================================================================
    # Evaluate
    # =====================================================================
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("RMSE:", rmse)

    # =====================================================================
    # Log ke MLflow
    # =====================================================================
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    print("Model berhasil dilog ke MLflow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)

    args = parser.parse_args()
    run_model(args.data_path, args.target_column)
