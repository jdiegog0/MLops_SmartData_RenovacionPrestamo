"""prepare_data.py — Carga y prepara el dataset de Renovación de Préstamo.

Este módulo carga el CSV original, aplica limpieza, imputación de nulos,
transformaciones logarítmicas y encoding de variables categóricas.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ruta absoluta basada en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Dataset_Renovacion_prestamo.csv"
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PREPARE | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Columnas originales que serán transformadas con logaritmo
COLS_LOG = [
    "Uso_Linea",
    "Uso_TrimLinea",
    "Saldo_Consumo",
    "SUELDO_ESTIMADO",
    "ANTIGUEDAD_MES",
    "Linea_Renovado",
    "Ahorro",
    "Prestamo_vigente",
    "Promed_6Mdeuda",
    "Deuda_Cubierta%",
]

# Columnas categóricas
CATEGORICAL_COLS = ["REGION", "SEXO", "EST_CIVIL"]

TARGET = "FLAG_VENTA"


def load_raw(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset crudo desde el CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}")
    df = pd.read_csv(data_path, sep=";")
    log.info("Dataset cargado correctamente. Shape original: %s", df.shape)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica los nombres de las columnas a un formato CamelCase/Standard."""
    rename_dict = {
        "LINEA_RENOVADO": "Linea_Renovado",
        "PLAZO_RENOVADO": "Plazo_Renovado",
        "USO_LINEA_TOTAL_TC_T2": "Uso_Linea",
        "USO_TRIM_LINEA_BBVA": "Uso_TrimLinea",
        "NR_ENTIDADES_TOTAL_T2": "Nro_Entidades",
        "DIFF_NRO_ENTIDA_TOTALES_T2_T12": "Diff_Nro_Entidades",
        "SDO_CONSUMO_T2": "Saldo_Consumo",
        "RESENCIA_OFERTA_PLD_RENOVADO": "Recencia_Oferta",
        "Ahorro_Sldo_Bco_T1": "Ahorro",
        "PConsumo_Sldo_Bco_T1": "Prestamo_vigente",
        "SDO_BCO_tot_sm_pasivo_Bco_6M": "Promed_6Mdeuda",
        "FLAG_LIMA_PROVINCIA": "Flag_Lima_Provincia",
        "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": "Deuda_Cubierta%",
    }
    df = df.rename(columns=rename_dict)
    return df


def cap_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los valores negativos en columnas financieras lógicas reemplazándolos por 0."""
    cols_to_cap = ["Ahorro", "Prestamo_vigente", "Promed_6Mdeuda", "SUELDO_ESTIMADO", "Deuda_Cubierta%"]
    for col in cols_to_cap:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la transformación logarítmica log1p para reducir el sesgo distributivo."""
    for col in COLS_LOG:
        if col in df.columns:
            df[f"{col}_LOG"] = np.log1p(df[col])
    return df


def impute_nulls(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Imputa los valores nulos detectados en variables numéricas y categóricas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Corrección para compatibilidad hacia atrás y adelante con Pandas 3.x/4.x strings
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    remaining_nulls = df.isnull().sum().sum()
    log.info("Imputación completada. Nulos restantes: %d", remaining_nulls)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica one-hot encoding a las variables categóricas."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)
    log.info("Encoding aplicado. Shape final: %s", df.shape)
    return df


def drop_original_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas numéricas originales que ya fueron transformadas a LOG."""
    cols_to_drop = [c for c in COLS_LOG if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def prepare(data_path: Path = DATA_PATH, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Pipeline completo de preparación de datos. Retorna df listo para modelar."""
    df = load_raw(data_path)
    df = rename_columns(df)
    df = cap_negative_values(df)
    df = apply_log_transform(df)
    df = impute_nulls(df, random_state=random_state)
    df = encode_categoricals(df)
    df = drop_original_numeric_cols(df)
    log.info("Preparación completada. Shape final: %s", df.shape)
    return df


if __name__ == "__main__":
    prepare()
    