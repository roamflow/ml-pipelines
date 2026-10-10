from ml_pipelines.assets.iris import iris_data, train_and_register_iris_model


def test_iris_data():
    data = iris_data()
    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data
    assert len(data["X_train"]) == 120
    assert len(data["X_test"]) == 30


def test_train_and_register_iris_model(tmp_path, monkeypatch):
    data = iris_data()
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    monkeypatch.setenv("MODEL_NAME", "test-iris-classifier")

    result = train_and_register_iris_model(data)

    assert result is not None
    assert "test_accuracy" in result.metadata
    assert "mlflow_run_id" in result.metadata
    assert result.metadata["test_accuracy"].value >= 0.8
