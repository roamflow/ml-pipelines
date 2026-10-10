import os

import mlflow
from dagster import MaterializeResult, MetadataValue, asset
from mlflow import MlflowClient
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@asset(
    description="Loads the Iris flower dataset and splits into train/test sets.",
    compute_kind="pandas",
    group_name="training",
)
def iris_data() -> dict:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@asset(
    description="Trains LogisticRegression on Iris, evaluates against Quality Gate (min accuracy), and promotes to 'Production' alias only if gate passes.",
    compute_kind="sklearn",
    group_name="training",
)
def train_and_register_iris_model(iris_data: dict) -> MaterializeResult:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    mlflow.set_experiment(experiment_name)

    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    max_iter = int(os.getenv("MAX_ITER", "200"))
    c_val = float(os.getenv("C_VALUE", "1.0"))
    min_acc_threshold = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.90"))

    X_train = iris_data["X_train"]
    X_test = iris_data["X_test"]
    y_train = iris_data["y_train"]
    y_test = iris_data["y_test"]

    model = LogisticRegression(max_iter=max_iter, C=c_val, random_state=42)
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, preds_train))
    test_acc = float(accuracy_score(y_test, preds_test))
    f1 = float(f1_score(y_test, preds_test, average="weighted"))

    with mlflow.start_run(run_name="dagster-iris-train") as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": max_iter,
                "C": c_val,
                "min_acc_threshold": min_acc_threshold,
            }
        )
        mlflow.log_metrics(
            {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "f1_score": f1,
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        run_id = run.info.run_id

    # Model Evaluation Gate
    passed_gate = test_acc >= min_acc_threshold
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_model = client.get_registered_model(model_name)
    latest_version = (
        registered_model.latest_versions[-1].version
        if registered_model.latest_versions
        else "1"
    )

    if passed_gate:
        # Promote winning model to 'Production' alias in MLflow 3.17 Registry
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=latest_version,
        )
        model_alias = "Production"
    else:
        # Keep existing production model intact; mark this candidate as rejected
        client.set_registered_model_alias(
            name=model_name,
            alias="Candidate_Rejected",
            version=latest_version,
        )
        model_alias = "Candidate_Rejected"

    return MaterializeResult(
        metadata={
            "model_name": MetadataValue.text(model_name),
            "model_version": MetadataValue.text(str(latest_version)),
            "model_alias": MetadataValue.text(model_alias),
            "evaluation_gate_passed": MetadataValue.bool(passed_gate),
            "min_accuracy_threshold": MetadataValue.float(min_acc_threshold),
            "test_accuracy": MetadataValue.float(test_acc),
            "train_accuracy": MetadataValue.float(train_acc),
            "f1_score": MetadataValue.float(f1),
            "mlflow_run_id": MetadataValue.text(run_id),
            "mlflow_model_uri": MetadataValue.text(
                f"models:/{model_name}@{model_alias}"
            ),
        }
    )


@asset(
    description="Compares baseline training features against test/serving features using Kolmogorov-Smirnov test to detect data distribution drift.",
    compute_kind="scipy",
    group_name="monitoring",
)
def detect_data_drift(iris_data: dict) -> MaterializeResult:
    X_train = iris_data["X_train"]
    X_test = iris_data["X_test"]

    p_value_threshold = float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05"))
    drift_results = {}
    drifted_features = []

    for col in X_train.columns:
        stat, p_val = ks_2samp(X_train[col], X_test[col])
        is_drifted = bool(p_val < p_value_threshold)
        drift_results[col] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": is_drifted,
        }
        if is_drifted:
            drifted_features.append(col)

    has_drift = len(drifted_features) > 0

    return MaterializeResult(
        metadata={
            "drift_detected": MetadataValue.bool(has_drift),
            "p_value_threshold": MetadataValue.float(p_value_threshold),
            "drifted_features_count": MetadataValue.int(len(drifted_features)),
            "drifted_features": MetadataValue.text(
                ", ".join(drifted_features) if drifted_features else "None"
            ),
            "feature_metrics": MetadataValue.json(drift_results),
            "status": MetadataValue.text(
                "ALERT: Data drift detected across features!"
                if has_drift
                else "OK: Data distributions match training baseline."
            ),
        }
    )
