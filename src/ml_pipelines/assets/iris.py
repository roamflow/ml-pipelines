import os

import mlflow
from dagster import MaterializeResult, MetadataValue, asset
from mlflow import MlflowClient
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
    description="Trains LogisticRegression on Iris, logs runs and registers model in MLflow Registry with 'Production' alias.",
    compute_kind="sklearn",
    group_name="training",
)
def train_and_register_iris_model(iris_data: dict) -> MaterializeResult:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    mlflow.set_experiment(experiment_name)

    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    max_iter = int(os.getenv("MAX_ITER", "200"))
    c_val = float(os.getenv("C_VALUE", "1.0"))

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
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": max_iter,
            "C": c_val,
        })
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "f1_score": f1,
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        run_id = run.info.run_id

    # Promote registered model to 'Production' alias in MLflow 3.17 Registry
    client = MlflowClient(tracking_uri=tracking_uri)
    # Find latest version for this model
    registered_model = client.get_registered_model(model_name)
    latest_version = registered_model.latest_versions[-1].version if registered_model.latest_versions else "1"

    # In MLflow 3+, aliases (e.g. @Production) supersede legacy stages
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=latest_version,
    )

    return MaterializeResult(
        metadata={
            "model_name": MetadataValue.text(model_name),
            "model_version": MetadataValue.text(str(latest_version)),
            "model_alias": MetadataValue.text("Production"),
            "test_accuracy": MetadataValue.float(test_acc),
            "train_accuracy": MetadataValue.float(train_acc),
            "f1_score": MetadataValue.float(f1),
            "mlflow_run_id": MetadataValue.text(run_id),
            "mlflow_model_uri": MetadataValue.text(f"models:/{model_name}@Production"),
        }
    )
