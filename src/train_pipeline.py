"""train_pipeline.py — Entrena el modelo de Renovación de Préstamo y guarda artefactos.

Pipeline:
  1. Carga y prepara el dataset (prepare_data.py)
  2. Split train/test estratificado
  3. Undersampling sobre el conjunto de entrenamiento (desbalanceo 96/4)
  4. GridSearchCV con RandomForestClassifier
  5. Evaluación con métricas AUC, Recall, F1 y Accuracy
  6. Serializa modelo (artifacts/modelo.pkl) y métricas (artifacts/metrics.json)
"""
import json
import pickle
import logging
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
)

from prepare_data import prepare, DATA_PATH, TARGET

# Rutas unificadas absolutas respecto al espacio de trabajo
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30
RECALL_MIN = 0.50   # quality gate — recall mínimo sobre la clase 1 (renovadores)

COLS_EXCLUDE = ["FLAG_VENTA", "MES", "CLIENTE"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN   | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_features(df: pd.DataFrame) -> list:
    """Devuelve la lista de variables predictoras excluyendo identificadores y el target."""
    return [col for col in df.columns if col not in COLS_EXCLUDE]


def perform_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE) -> tuple:
    """Aplica undersampling aleatorio controlado para manejar la asimetría de clases (96/4)."""
    df_temp = X.copy()
    df_temp[TARGET] = y

    df_clase_1 = df_temp[df_temp[TARGET] == 1]
    df_clase_0 = df_temp[df_temp[TARGET] == 0]

    n_samples_0 = len(df_clase_1) * 3  # Proporción aproximada 75/25 objetivo
    if n_samples_0 > len(df_clase_0):
        n_samples_0 = len(df_clase_0)

    df_clase_0_sub = df_clase_0.sample(n=n_samples_0, random_state=random_state)
    df_balanced = pd.concat([df_clase_1, df_clase_0_sub], axis=0).sample(frac=1, random_state=random_state)

    return df_balanced.drop(columns=[TARGET]), df_balanced[TARGET]


def train(df: pd.DataFrame) -> dict:
    """Ejecuta el ciclo completo de optimización, entrenamiento y persistencia."""
    features = get_features(df)
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    log.info("Dataset original: %s | Train: %s | Test: %s", X.shape[0], X_train.shape[0], X_test.shape[0])
    X_train_u, y_train_u = perform_undersampling(X_train, y_train, random_state=RANDOM_STATE)
    log.info("Dataset balanceado (Train Undersampled): %s", X_train_u.shape[0])

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [6, 10, 15],
        "min_samples_leaf": [2, 5],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    gs = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=StratifiedKFold(3),
        n_jobs=-1,
        verbose=1,
    )
    log.info("Iniciando GridSearchCV...")
    gs.fit(X_train_u, y_train_u)
    log.info("Mejores params: %s", gs.best_params_)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metricas = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "params": gs.best_params_,
        "recall_minimo": RECALL_MIN,
        "features": features,
    }

    log.info(
        "ROC-AUC=%.4f | Recall=%.4f | F1=%.4f | Acc=%.4f",
        metricas["roc_auc"],
        metricas["recall"],
        metricas["f1"],
        metricas["accuracy"],
    )

    ARTIFACTS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    log.info("Modelo guardado exitosamente en -> %s", MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metricas, f, indent=4)
    log.info("Métricas guardadas exitosamente en -> %s", METRICS_PATH)

    return metricas


if __name__ == "__main__":
    df_prepared = prepare(DATA_PATH)
    train(df_prepared)