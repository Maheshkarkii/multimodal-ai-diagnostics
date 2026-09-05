"""
Unit and integration tests for Phase 4 Sensor Intelligence and Anomaly Modeling.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sensor.analysis.feature_importance import compute_permutation_feature_importance
from src.sensor.data.generate_sample_telemetry import generate_synthetic_telemetry_dataset
from src.sensor.data.sensor_dataset import SensorDataValidator
from src.sensor.inference.sensor_predictor import SensorPredictor
from src.sensor.models.anomaly_detector import SensorAnomalyDetector
from src.sensor.models.sensor_mlp import build_sensor_model
from src.sensor.preprocessing.sensor_scaler import SensorPreprocessor


def test_sensor_preprocessor_leakage_isolation():
    feature_cols = ["temp", "vib"]
    df_train = pd.DataFrame({"temp": [10.0, 20.0, np.nan], "vib": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"temp": [30.0, np.nan], "vib": [4.0, 5.0]})

    preprocessor = SensorPreprocessor(feature_columns=feature_cols)
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    # Impute value for temp must be median of train ([10, 20] -> 15.0)
    assert preprocessor.impute_values["temp"] == 15.0
    assert X_train.shape == (3, 2)
    assert X_test.shape == (2, 2)


def test_sensor_mlp_forward_and_embedding():
    model = build_sensor_model(in_features=6, num_classes=5, embedding_dim=256)
    dummy_input = torch.randn(4, 6)

    logits, embeddings = model(dummy_input, return_features=True)
    assert logits.shape == (4, 5)
    assert embeddings.shape == (4, 256)

    standalone_emb = model.extract_features(dummy_input)
    assert standalone_emb.shape == (4, 256)


def test_anomaly_detector_scoring():
    feature_names = ["temp", "vib"]
    X_normal = np.random.normal(loc=[50.0, 2.0], scale=[2.0, 0.2], size=(100, 2))

    detector = SensorAnomalyDetector(feature_names=feature_names, contamination=0.05)
    detector.fit(X_normal)

    # Normal sample
    norm_res = detector.evaluate_sample(np.array([50.5, 2.1]))
    assert 0.0 <= norm_res["anomaly_score"] <= 1.0

    # Extreme anomaly
    anom_res = detector.evaluate_sample(np.array([120.0, 15.0]))
    assert anom_res["anomaly_score"] > norm_res["anomaly_score"]


def test_telemetry_generator_and_validator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_telemetry.csv"
        generate_synthetic_telemetry_dataset(csv_path, num_machines=4, records_per_machine=20, seed=42)

        df = pd.read_csv(csv_path)
        feature_cols = [
            "temperature_c",
            "vibration_rms_g",
            "rotational_speed_rpm",
            "motor_current_a",
            "hydraulic_pressure_bar",
            "load_percentage",
        ]
        summary = SensorDataValidator.validate_and_summarize(df, feature_cols, "fault_label", "machine_id")

        assert summary["total_rows"] == 80
        assert summary["unique_machines"] == 4
        assert len(summary["class_distribution"]) == 5


def test_permutation_importance_execution():
    model = build_sensor_model(in_features=3, num_classes=2, embedding_dim=64)
    X_val = np.random.randn(20, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, size=20)

    importances = compute_permutation_feature_importance(
        model, X_val, y_val, feature_names=["f1", "f2", "f3"], device="cpu"
    )
    assert len(importances) == 3
    assert abs(sum(importances.values()) - 100.0) < 1.0


def test_sensor_predictor_direct_inference():
    feature_cols = ["f1", "f2"]
    preprocessor = SensorPreprocessor(feature_columns=feature_cols)
    df_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    preprocessor.fit(df_train)

    model = build_sensor_model(in_features=2, num_classes=3, embedding_dim=256)
    detector = SensorAnomalyDetector(feature_names=feature_cols).fit(df_train.to_numpy())

    predictor = SensorPredictor(
        model=model,
        preprocessor=preprocessor,
        anomaly_detector=detector,
        class_names=["Normal", "Fault_A", "Fault_B"],
        device="cpu",
    )

    res = predictor.predict({"f1": 2.5, "f2": 25.0}, top_k=2, return_embedding=True)
    assert "predicted_machine_state" in res
    assert "anomaly_assessment" in res
    assert res["embedding_dim"] == 256
    assert len(res["sensor_embedding"]) == 256
