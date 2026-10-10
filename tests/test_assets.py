from ml_pipelines.assets.iris import (
    detect_data_drift,
    iris_data,
    train_and_register_iris_model,
)


def test_iris_data():
    data = iris_data()
    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data
    assert len(data["X_train"]) == 120
    assert len(data["X_test"]) == 30


def test_train_and_register_iris_model_passes_gate(tmp_path, monkeypatch):
    data = iris_data()
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    monkeypatch.setenv("MODEL_NAME", "test-iris-classifier")
    monkeypatch.setenv("MIN_ACCURACY_THRESHOLD", "0.85")

    result = train_and_register_iris_model(data)

    assert result is not None
    assert result.metadata["evaluation_gate_passed"].value is True
    assert result.metadata["model_alias"].value == "Production"
    assert result.metadata["test_accuracy"].value >= 0.85


def test_train_and_register_iris_model_fails_gate(tmp_path, monkeypatch):
    data = iris_data()
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    monkeypatch.setenv("MODEL_NAME", "test-iris-classifier")
    # Impossibly high threshold ensures evaluation gate rejects promotion
    monkeypatch.setenv("MIN_ACCURACY_THRESHOLD", "0.9999")

    result = train_and_register_iris_model(data)

    assert result is not None
    assert result.metadata["evaluation_gate_passed"].value is False
    assert result.metadata["model_alias"].value == "Candidate_Rejected"


def test_detect_data_drift_clean():
    data = iris_data()
    result = detect_data_drift(data)
    assert result is not None
    assert result.metadata["drift_detected"].value is False
    assert result.metadata["drifted_features_count"].value == 0


def test_detect_data_drift_shifted():
    data = iris_data()
    # Artificially shift test distribution by adding a large offset
    shifted_data = {
        "X_train": data["X_train"],
        "X_test": data["X_test"] + 100.0,
        "y_train": data["y_train"],
        "y_test": data["y_test"],
    }
    result = detect_data_drift(shifted_data)
    assert result is not None
    assert result.metadata["drift_detected"].value is True
    assert result.metadata["drifted_features_count"].value > 0
