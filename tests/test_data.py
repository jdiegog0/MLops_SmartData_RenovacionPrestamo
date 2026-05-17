"""tests/test_data.py — Tests del módulo de preparación de datos."""
import sys
import pandas as pd
##import numpy as np
##import pytest

sys.path.insert(0, "src")
from prepare_data import (  # noqa: E402
    load_raw,
    rename_columns,
    cap_negative_values,
    apply_log_transform,
    impute_nulls,
    encode_categoricals,
    prepare,
    DATA_PATH,
    TARGET,
    COLS_LOG,
)


def test_load_raw_returns_dataframe():
    """load_raw() debe retornar un DataFrame."""
    df = load_raw(DATA_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_raw_has_rows():
    """El dataset debe tener filas."""
    df = load_raw(DATA_PATH)
    assert len(df) > 0


def test_load_raw_has_target_column():
    """El dataset crudo debe tener la columna FLAG_VENTA."""
    df = load_raw(DATA_PATH)
    assert TARGET in df.columns


def test_rename_columns_maps_correctly():
    """rename_columns() debe renombrar LINEA_RENOVADO → Linea_Renovado."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    assert "Linea_Renovado" in df.columns
    assert "LINEA_RENOVADO" not in df.columns


def test_cap_negative_values():
    """cap_negative_values() debe eliminar negativos en Ahorro, Prestamo_vigente."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    df = cap_negative_values(df)
    for col in ["Ahorro", "Prestamo_vigente", "Promed_6Mdeuda"]:
        if col in df.columns:
            assert (df[col] >= 0).all(), f"Hay negativos en {col}"


def test_log_transform_creates_new_cols():
    """apply_log_transform() debe crear columnas _LOG."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    df = apply_log_transform(df)
    for col in COLS_LOG:
        new_col = f"{col}_LOG"
        if col in df.columns:
            assert new_col in df.columns, f"Falta columna {new_col}"


def test_log_transform_no_negatives():
    """Las columnas _LOG no deben tener valores negativos."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    df = cap_negative_values(df)
    df = apply_log_transform(df)
    log_cols = [f"{c}_LOG" for c in COLS_LOG if c in df.columns]
    for col in log_cols:
        non_null = df[col].dropna()
        assert (non_null >= 0).all(), f"Negativos en {col}"


def test_impute_nulls_reduces_nulls():
    """impute_nulls() debe reducir o eliminar los nulos en columnas clave."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    df = cap_negative_values(df)
    df = apply_log_transform(df)
    nulls_before = df.isnull().sum().sum()
    df = impute_nulls(df)
    nulls_after = df.isnull().sum().sum()
    assert nulls_after <= nulls_before


def test_encode_categoricals_no_object_cols():
    """encode_categoricals() no debe dejar columnas de tipo object."""
    df = load_raw(DATA_PATH)
    df = rename_columns(df)
    df = cap_negative_values(df)
    df = apply_log_transform(df)
    df = impute_nulls(df)
    df = encode_categoricals(df)
    object_cols = df.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Quedan columnas object: {object_cols}"


def test_target_is_binary():
    """FLAG_VENTA debe ser binario (0 o 1)."""
    df = prepare(DATA_PATH)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_target_class_imbalance():
    """La tasa de clase 1 (renovación) debe estar entre 2% y 15%."""
    df = prepare(DATA_PATH)
    rate = df[TARGET].mean()
    assert 0.02 <= rate <= 0.15, f"Tasa de clase 1 fuera de rango: {rate:.2%}"


def test_prepare_no_nulls_in_key_cols():
    """El dataset preparado no debe tener nulos en columnas LOG."""
    df = prepare(DATA_PATH)
    log_cols = [c for c in df.columns if "_LOG" in c]
    for col in log_cols:
        assert df[col].isnull().sum() == 0, f"Nulos en {col}"


def test_prepare_returns_dataframe():
    """prepare() debe retornar un DataFrame."""
    df = prepare(DATA_PATH)
    assert isinstance(df, pd.DataFrame)


def test_prepare_has_more_than_10_cols():
    """El dataset preparado debe tener más de 10 columnas."""
    df = prepare(DATA_PATH)
    assert len(df.columns) > 10
