"""validate_model.py — Valida que las métricas superen el umbral mínimo (quality gate).

Si falla, el pipeline CI/CD se detiene y no se construye la imagen Docker.
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


def validate(metrics_path: Path = METRICS_PATH) -> None:
    """Valida métricas. Lanza SystemExit(1) si falla el quality gate."""
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} no existe. Ejecuta train_pipeline.py primero.")
        sys.exit(1)

    with open(metrics_path) as f:
        m = json.load(f)

    umbral_recall = m.get("recall_minimo", 0.50)
    recall = m.get("recall", 0.0)
    f1 = m.get("f1", 0.0)
    roc_auc = m.get("roc_auc", 0.0)
    accuracy = m.get("accuracy", 0.0)
    precision = m.get("precision", 0.0)

    print("=" * 55)
    print(" QUALITY GATE — RENOVACIÓN DE PRÉSTAMO")
    print("=" * 55)
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Recall    : {recall:.4f}  (umbral mínimo: >= {umbral_recall})")
    print(f"  Precision : {precision:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print("=" * 55)

    if recall < umbral_recall:
        print(f"❌ FALLA: El Recall ({recall:.4f}) es inferior al umbral mínimo ({umbral_recall}).")
        sys.exit(1)

    print("✅ ¡ÉXITO!: El modelo supera satisfactoriamente todos los criterios establecidos.")
    sys.exit(0)


if __name__ == "__main__":
    validate()