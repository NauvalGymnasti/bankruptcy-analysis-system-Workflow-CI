import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor

def load_data(path):
    df = pd.read_csv(path)
    return df

def train(data_path, n_estimators, max_depth, learning_rate, subsample):

    # MLflow auto logging
    mlflow.autolog()

    # Load data
    df = load_data(data_path)

    X = df.drop("ihsg_terakhir", axis=1)
    y = df["ihsg_terakhir"]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42
    )

    model.fit(X, y)

    # Simpan model
    mlflow.sklearn.log_model(model, "xgb_model")

    print("Training selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample
    )
