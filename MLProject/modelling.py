import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# FORCE LOCAL MLFLOW (AMAN CI)
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telecom-churn-experiment")

# =========================
# PATH CONFIG (RELATIVE ONLY)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "telecom_churn_preprocessing.csv")
TARGET_VAR = "Churn"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_VAR])
y = df[TARGET_VAR]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN + LOGGING
# =========================
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # LOG PARAM & METRIC
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # LOG MODEL
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.4f}")