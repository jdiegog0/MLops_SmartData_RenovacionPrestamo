# Pipeline End-to-End de MLOps para la Clasificación de Renovación de Préstamos (`RenvPr`)

Este repositorio contiene el pipeline robusto de Machine Learning y la arquitectura de MLOps estructurada para abordar el problema estratégico de **Renovación Anticipada de Préstamos**. El proyecto implementa un ciclo automatizado completo: desde la ingesta y preparación de datos estructurados, optimización hiperparamétrica automatizada, pruebas unitarias integradas, validación estricta de métricas (*Quality Gates*) hasta la contenerización con Docker para su despliegue seguro.

---

## 1. Contexto de Negocio y Objetivos

### El Problema Comercial
La entidad financiera ha identificado una ventana de oportunidad crítica: la renovación de préstamos vigentes es altamente efectiva si la entidad contacta proactivamente al cliente **antes que las financieras competidoras**. El mercado de créditos es altamente agresivo y el costo de adquisición de un cliente nuevo supera drásticamente el costo de retención y recompra de la cartera actual.

### Objetivo del Proyecto
A partir de la información histórica recopilada durante **2 meses de campañas de ventas vía Call Center**, cruzada con el comportamiento financiero interno de los clientes en la entidad, el objetivo es:
1. **Modelar la propensión**: Construir un modelo predictivo capaz de identificar con alta sensibilidad (Recall) qué clientes tienen la mayor probabilidad y perfil de interés para aceptar una renovación de préstamo.
2. **Optimización del Contacto**: Permitir al equipo de Marketing segmentar y priorizar los esfuerzos del Call Center, mitigando la saturación de canales y anticipándose a la competencia en el momento oportuno.

---

## 2. Arquitectura del Repositorio y Componentes

La estructura sigue un patrón limpio y modular preparado para entornos de producción y metodologías de CI/CD:

```text
RenvPr/
├── .pytest_cache/             # Caché local de ejecución de pruebas
├── artifacts/                 # Artefactos generados por el pipeline (Excluidos en .gitignore)
│   ├── modelo.pkl             # Modelo RandomForestClassifier serializado
│   └── metrics.json           # Métricas de evaluación y parámetros del GridSearch
├── data/                      # Datos del proyecto (Locales o montados)
│   └── Dataset_Renovacion_prestamo.csv
├── src/                       # Código fuente del pipeline
│   ├── __init__.py
│   ├── prepare_data.py        # ETL, Limpieza, Imputación, Transformaciones LOG y Encoding
│   ├── train_pipeline.py      # Split Estratificado, Undersampling, GridSearchCV y Persistencia
│   └── validate_model.py      # Evaluador automático del Quality Gate de métricas
├── tests/                     # Suite de Pruebas Unitarias Integradas
│   ├── __init__.py
│   ├── test_data.py           # Validaciones del módulo de datos y transformaciones
│   ├── test_model.py          # Validaciones de consistencia y firmas del modelo entrenado
│   └── test_pipeline.py       # Validaciones funcionales de extremo a extremo del entrenamiento
├── Dockerfile                 # Contenerización aislada de la solución para producción
├── Makefile                   # Automatización unificada del Pipeline local y de CI/CD
└── setup.cfg                  # Configuración centralizada de linters (flake8) y pytest