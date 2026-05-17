"""tests/test_pipeline.py — Tests del pipeline de entrenamiento."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, "src")
from prepare_data import prepare, DATA_PATH  # noqa: E402
from train_pipeline import train  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "modelo.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


@pytest.fixture(scope="module")
def ejecutar_pipeline_real():
    """Fixture que asegura que las métricas en memoria estén disponibles."""
    df = prepare(DATA_PATH)
    return train(df)


def test_train_returns_metrics(ejecutar_pipeline_real):
    """train() debe retornar un dict con las métricas esperadas."""
    metricas = ejecutar_pipeline_real
    assert "f1" in metricas
    assert "recall" in metricas
    assert "accuracy" in metricas
    assert "roc_auc" in metricas


def test_train_f1_positive(ejecutar_pipeline_real):
    """El F1-score debe ser mayor que 0."""
    assert ejecutar_pipeline_real["f1"] > 0.0


def test_train_metrics_in_range(ejecutar_pipeline_real):
    """Las métricas deben estar en el rango [0, 1]."""
    for key in ["f1", "recall", "accuracy", "roc_auc"]:
        assert 0.0 <= ejecutar_pipeline_real[key] <= 1.0


def test_train_saves_model():
    """Verifica que artifacts/modelo.pkl exista en la raíz del proyecto."""
    assert MODEL_PATH.exists()


def test_train_saves_metrics():
    """Verifica que artifacts/metrics.json exista en la raíz del proyecto."""
    assert METRICS_PATH.exists()


def test_train_saves_best_params():
    """metrics.json debe incluir los mejores parámetros del GridSearch."""
    with open(METRICS_PATH) as f:
        m = json.load(f)
    assert "params" in m
    assert isinstance(m["params"], dict)
    