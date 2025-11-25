import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def load_data(path):
    return pd.read_csv(path)

def train_model(df):
    X = df.iloc[:, :-1]   # semua kolom kecuali target
    y = df.iloc[:, -1]    # kolom terakhir sbg target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    print("Model Score (R2):", score)
    return model, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        # Load data
        df = load_data(args.data_path)

        # Train
        model, r2 = train_model(df)

        # Log metric
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.xgboost.log_model(model, "model")
