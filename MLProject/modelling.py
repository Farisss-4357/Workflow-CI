import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# =========================================================================
# === KONFIGURASI PATH (KOREKSI MUTLAK) ===
# Path dikembalikan ke NAMA FILE SAJA. 
# Ini karena perintah 'mlflow run MLProject/' membuat direktori kerja berada di dalam MLProject/.
TRAIN_FILE = os.path.join("train_data.csv")
TEST_FILE = os.path.join("test_data.csv")
# =========================================================================

TARGET_COLUMN = "species"

mlflow.sklearn.autolog() # Autologging tetap dipertahankan

def load_data():
    """Memuat data training dan testing yang sudah dipisahkan."""
    try:
        print(f"Mencoba memuat data latih dari: {TRAIN_FILE}")
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)

        # Pisahkan fitur (X) dan target (y)
        X_train = train_df.drop(TARGET_COLUMN, axis=1)
        y_train = train_df[TARGET_COLUMN]
        
        X_test = test_df.drop(TARGET_COLUMN, axis=1)
        y_test = test_df[TARGET_COLUMN]
        
        print("Data training dan testing berhasil dimuat dan dipisahkan.")
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError as e:
        print("==================================================")
        print("ERROR KRITIS: FILE DATA TIDAK DITEMUKAN.")
        print(f"PATH YANG DICARI: {e.filename}")
        print("Pastikan file data ada di folder MLProject/ dan path di skrip sudah benar.")
        print("==================================================")
        return None, None, None, None

def train_model_basic():
    """Melatih model Logistic Regression dan mencatatnya ke MLflow menggunakan Autolog."""
    
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None:
        return
    # 1. Inisialisasi Model dan Hyperparameter
    model = LogisticRegression(
        solver='liblinear',
        max_iter=500,
        random_state=42
    )
    
    # 2. Melatih Model (model.fit)
    print("\nMemulai pelatihan model...")
    model.fit(X_train, y_train)
    print("Pelatihan model selesai. Detail dicatat oleh MLflow.")

    # 3. Evaluasi dan Verifikasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Catat metrik kustom
    mlflow.log_metric("final_test_accuracy", acc)
    print(f"Final Test Accuracy (Metrik Kustom): {acc:.4f}")
    
    print("\n---------------------------------------------------------")
    print("Verifikasi di MLflow UI untuk melihat Model, Params, dan Metrics.")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    train_model_basic()