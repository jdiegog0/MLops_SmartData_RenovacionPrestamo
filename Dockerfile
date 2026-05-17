# Dockerfile — Modelo de Renovación de Préstamo
# Usado por GitHub Actions y por make docker

FROM python:3.10-slim

LABEL maintainer="MLOps Renovacion Prestamo"
LABEL description="Imagen CI/CD del modelo de clasificación de renovación de préstamos"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
# IMPORTANTE: copiar requirements.txt antes del código para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Copiar datos
COPY data/ ./data/

# Copiar artefactos del modelo (generados por train_pipeline.py)
# En GitHub Actions estos llegan via download-artifact
COPY artifacts/ ./artifacts/

# Verificación rápida al construir
RUN python -c "import pickle; \
    m = pickle.load(open('artifacts/modelo.pkl', 'rb')); \
    print('Modelo cargado OK:', type(m).__name__)" || echo "artifacts/ no disponible en build"

EXPOSE 8000

# Por defecto, verificar que el modelo carga correctamente
CMD ["python", "-c", \
     "import pickle; m=pickle.load(open('artifacts/modelo.pkl','rb')); \
      print('Modelo listo:', type(m).__name__, '| n_estimators:', m.n_estimators)"]
