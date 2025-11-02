# Data & Visualisasi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pra-pemrosesan & Model ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("dataset/Company_Bankruptcy_Prediction_data.csv")
    
    # ====================================================
    # 1. Identifikasi dan Tangani Nilai Placeholder (9990000000 / 10000000000)
    # ====================================================
    # Nilai seperti ini biasanya representasi missing value dalam dataset keuangan
    placeholder_values = [9990000000, 10000000000, 999000000, 1000000000]
    df = df.replace(placeholder_values, np.nan)

    print("\nJumlah missing value setelah mengganti placeholder:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # ====================================================
    # 2. Hapus Fitur Konstan / Hampir Konstan
    # ====================================================
    # Kolom dengan variasi (std) < 1e-5 dianggap tidak informatif
    low_variance_cols = [col for col in df.columns if df[col].std() < 1e-5]
    print(f"\nJumlah fitur konstan/hampir konstan: {len(low_variance_cols)}")

    # Hapus fitur tersebut
    df.drop(columns=low_variance_cols, inplace=True)

    # ====================================================
    # 3. Pisahkan Fitur dan Target
    # ====================================================
    target_col = 'Bankrupt?'  # sesuaikan dengan nama kolom target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ====================================================
    # 4. Imputasi Missing Values
    # ====================================================
    # Gunakan median karena dataset berisi rasio keuangan dengan banyak outlier
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ====================================================
    # 5. Scaling (RobustScaler)
    # ====================================================
    # RobustScaler cocok karena tahan terhadap outlier ekstrem
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # ====================================================
    # 🔀 7. Split Data Train & Test
    # ====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n✅ Data preprocessing selesai!")
    print(f"Ukuran data train: {X_train.shape}")
    print(f"Ukuran data test: {X_test.shape}")

    # ====================================================
    # 🔎 8. Validasi Akhir
    # ====================================================
    print("\nCek distribusi kelas pada train dan test:")
    print("Train:")
    print(y_train.value_counts(normalize=True))
    print("\nTest:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test
