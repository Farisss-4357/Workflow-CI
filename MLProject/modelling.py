import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===============================
# CONFIG
# ===============================
DATA_PATH = "telecom_churn_preprocessing.csv"
TARGET_COL = "Churn"

# ===============================
# LOAD DATA
# ===============================
def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# ===============================
# TRAIN MODEL
# ===============================
def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ===============================
    # METRICS
    # ===============================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ===============================
    # LOGGING (TANPA start_run)
    # ===============================
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("✅ Training selesai")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    train_model()
