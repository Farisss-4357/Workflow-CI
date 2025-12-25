import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
CSV_URL = os.getenv("CSV_URL", "train_pca.csv")
TARGET_VAR = os.getenv("TARGET_VAR", "Credit_Score")

# =========================
# TRACKING URI
# =========================
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("credit_scoring_model")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_URL)

X = df.drop(columns=[TARGET_VAR])
y = df[TARGET_VAR]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAINING
# =========================
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # =========================
    # LOGGING
    # =========================
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CreditScoringModel"
    )

    print(f"Accuracy: {acc}")