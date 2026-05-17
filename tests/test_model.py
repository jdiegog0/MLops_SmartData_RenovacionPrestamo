"""tests/test_model.py — Tests del modelo serializado."""
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, "src")
from train_pipeline import get_features  # noqa: E402
from prepare_data import prepare, DATA_PATH  # noqa: E402

# Localizar los artefactos reales del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "modelo.pkl"


@pytest.fixture(scope="module")
def modelo_entrenado():
    """Fixture: Carga el modelo real que ya fue generado por 'make train'."""
    if not MODEL_PATH.exists():
        pytest.fail(f"El modelo no existe en {MODEL_PATH}. Ejecuta 'make train' primero.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def features_list():
    """Fixture: devuelve la lista de features del dataset preparado."""
    df = prepare(DATA_PATH)
    return get_features(df)


def test_model_has_predict(modelo_entrenado):
    """El modelo debe tener el método predict."""
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    """El modelo debe tener el método predict_proba."""
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado, features_list):
    """predict debe retornar solo 0 o 1."""
    import pandas as pd
    n = 10
    rng = np.random.default_rng(42)
    sample = pd.DataFrame(
        rng.random((n, len(features_list))),
        columns=features_list,
    )
    y_pred = modelo_entrenado.predict(sample)
    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado, features_list):
    """predict_proba debe retornar shape (n, 2) con valores en [0, 1]."""
    import pandas as pd
    n = 10
    rng = np.random.default_rng(1)
    sample = pd.DataFrame(
        rng.random((n, len(features_list))),
        columns=features_list,
    )
    proba = modelo_entrenado.predict_proba(sample)
    assert proba.shape == (n, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado, features_list):
    """Las probabilidades por fila deben sumar 1."""
    import pandas as pd
    n = 15
    rng = np.random.default_rng(2)
    sample = pd.DataFrame(
        rng.random((n, len(features_list))),
        columns=features_list,
    )
    proba = modelo_entrenado.predict_proba(sample)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_is_random_forest(modelo_entrenado):
    """El modelo debe ser una instancia de RandomForestClassifier o similar."""
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(modelo_entrenado, RandomForestClassifier)


def test_model_feature_importances(modelo_entrenado, features_list):
    """El modelo debe exponer la importancia de las variables con el tamaño correcto."""
    assert hasattr(modelo_entrenado, "feature_importances_")
    assert len(modelo_entrenado.feature_importances_) == len(features_list)
    