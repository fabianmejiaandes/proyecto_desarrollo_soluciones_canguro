# Proyecto Desarrollo Soluciones Canguro

Repositorio del microproyecto para **predicción de malnutrición / crecimiento no armónico** a los **12 meses de edad corregida**, y un **dashboard** para visualización de indicadores.

## Aplicaciones del proyecto

- **Entrenamiento y evaluación de modelos (ML)**: `malnutrition_model.py`
- **Predicción / integración del modelo empaquetado**: `predict.py` (ver `MODEL_DOCUMENTATION.md`)
- **Dashboard (Dash)**: `dashboard/`

## Estructura del repositorio

```
.
├── malnutrition_model.py               # Entrenamiento, evaluación, export de modelo
├── predict.py                          # Predictor (carga artefactos y predice)
├── MODEL_DOCUMENTATION.md              # Guía de integración del predictor
├── dashboard/                          # App Dash (visualización)
├── data/                               # Punteros DVC a datasets (Excel)
├── notebooks/                          # Notebook de exploración
├── output/                             # Artefactos generados (métricas, figuras, package)
└── problematica/                       # PDF de referencia del microproyecto
```

## Requisitos

- Python 3.10+ recomendado
- (Opcional) **DVC** si vas a descargar los datasets definidos en `data/*.dvc`

## Ejecución local (paso a paso)

### 1) Crear entorno e instalar dependencias

Desde la raíz del repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install scikit-learn joblib pandas numpy matplotlib seaborn plotly openpyxl dash dash-bootstrap-components
```

> Nota: el dashboard también se puede instalar con `pip install -r dashboard/requirements.txt` (ver sección Dashboard).

### 2) Datos

Los datasets de `data/` están versionados con DVC y apuntan a un remoto S3 (ver `.dvc/config`).

- Si tienes acceso al remoto:

```bash
pip install dvc[s3]
dvc pull
```

- Si no tienes acceso, puedes ejecutar entrenamiento pasando rutas locales:
  - `--data-path <archivo_excel_datos>`
  - `--dict-path <archivo_excel_diccionario>`

### 3) Entrenar y comparar modelos

Genera métricas y artefactos de salida en `output/`:

```bash
python malnutrition_model.py --model all
```

Para entrenar modelos específicos:

```bash
python malnutrition_model.py --model logistic_regression random_forest gradient_boosting
```

### 4) Exportar (empaquetar) el mejor modelo para uso en predicción

```bash
python malnutrition_model.py --model gradient_boosting --export-model
```

Esto genera `output/model_package/` con:

- `model.joblib`
- `imputer.joblib`
- `pipeline_metadata.json`
- `feature_names.json`

### 5) Probar predicción (script)

Ejecuta:

```bash
python predict.py --package-dir output/model_package --input datos_paciente.json
```

La guía completa de integración está en `MODEL_DOCUMENTATION.md`.

### 6) Ejecutar Dashboard

```bash
pip install -r dashboard/requirements.txt
python dashboard/app.py
```

Abrir `http://127.0.0.1:8050/`.

## Despliegue en la nube (paso a paso)

Abajo se deja un flujo recomendado con **Render** (simple para Dash). Si usas otro proveedor (Railway, Heroku, AWS), los pasos son equivalentes: instalar dependencias y ejecutar `gunicorn` apuntando a `server`.

### A) Desplegar el Dashboard en Render

1. Subir el repo a GitHub.
2. En Render: **New +** → **Web Service** → conectar el repo.
3. Configurar:
   - **Root Directory**: `dashboard`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
4. Variables de entorno:
   - (Opcional) `PORT` (Render la define automáticamente).
5. Deploy y abrir el URL público.

### B) Desplegar el predictor como API (opcional)

El repo incluye `predict.py` como módulo/script. Si necesitas “aplicación” expuesta como servicio HTTP, lo más común es envolver `predict.py` con un microservicio (por ejemplo FastAPI) y desplegarlo como Web Service.

Si quieres, puedo dejarte un `api/` mínimo (FastAPI) con endpoint `/predict`, listo para desplegar en Render/Railway.

## Artefactos de salida (métricas, figuras, RF/GB)

Ver `output/README.md`.

