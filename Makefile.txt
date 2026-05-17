# Makefile — Pipeline CI/CD Local — Renovación de Préstamo
# Simula exactamente los mismos pasos que GitHub Actions antes de hacer push
# Uso: make <target>
# Uso completo: make all

.PHONY: all lint test train validate docker clean help

# ── Target por defecto: ejecutar todo en secuencia (Asegurando train antes de test) ──
all: lint train test validate docker
	@echo ""
	@echo "✓ Pipeline CI/CD local completado. Listo para git push."

# ── Linting con flake8 ────────────────────────────────────────────────────────
lint:
	@echo "=== Linting con flake8 ==="
	flake8 src/ tests/ --config=setup.cfg
	@echo "Linting OK"

# ── Entrenar modelo (preparación de datos + entrenamiento) ───────────────────
train:
	@echo "=== Entrenando modelo ==="
	python src/train_pipeline.py

# ── Tests unitarios con pytest ────────────────────────────────────────────────
test:
	@echo "=== Tests con pytest ==="
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Quality gate de métricas ──────────────────────────────────────────────────
validate:
	@echo "=== Validando métricas (quality gate) ==="
	python src/validate_model.py

# ── Build de imagen Docker ────────────────────────────────────────────────────
docker:
	@echo "=== Construyendo imagen Docker ==="
	docker build -t renovacion-prestamo:local .
	@echo "=== Smoke test ==="
	docker run --rm renovacion-prestamo:local python -c \
		"import pickle; m=pickle.load(open('artifacts/modelo.pkl','rb')); print('Modelo OK:', type(m).__name__)"

# ── Limpiar artefactos generados ──────────────────────────────────────────────
clean:
	rm -rf artifacts/ __pycache__ .coverage htmlcov/ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Targets disponibles:"
	@echo "  make all       — Ejecutar todo: lint + train + test + validate + docker"
	@echo "  make lint      — Solo linting con flake8"
	@echo "  make train     — Preparar datos + entrenar modelo"
	@echo "  make test      — Solo tests con pytest"
	@echo "  make validate  — Quality gate de métricas"
	@echo "  make docker    — Construir imagen Docker y correr smoke test"
	@echo "  make clean     — Remover archivos temporales y artefactos"