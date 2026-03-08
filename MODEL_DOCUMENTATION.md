# Documentacion del Modelo de Prediccion de Malnutricion Infantil

## Guia de integracion para `predict.py` - Fundacion Canguro

---

## Descripcion General

Modelo **Gradient Boosting** empaquetado para predecir el estado nutricional (indice de crecimiento armonico) a los **12 meses de edad corregida** en niños prematuros y/o de bajo peso al nacer.

- **Variable objetivo**: `indexnutricion12meses`
  - `0` = Crecimiento NO armonico (malnutricion / falla en crecimiento)
  - `1` = Crecimiento armonico (adecuado)

- **Metricas del modelo**:
  - F1-Score: 0.835
  - ROC-AUC: 0.813
  - Accuracy: 0.777
  - Balanced Accuracy: 0.735

---

## 1. Empaquetamiento del Modelo

El modelo debe ser entrenado y empaquetado ejecutando:

```bash
python malnutrition_model.py --model gradient_boosting --export-model
```

Esto genera la carpeta `output/model_package/` con los siguientes artefactos:

```
output/model_package/
    model.joblib              # Modelo Gradient Boosting entrenado
    imputer.joblib            # Imputer (mediana) ajustado al dataset de entrenamiento
    pipeline_metadata.json    # Metadatos del pipeline completo
    feature_names.json        # Orden exacto de features que espera el modelo
```

### Opciones adicionales de empaquetamiento

```bash
# Especificar directorio de exportacion
python malnutrition_model.py --model gradient_boosting --export-model --export-dir ./mi_modelo

# Con hiperparametros personalizados
python malnutrition_model.py --model gradient_boosting --export-model \
    --gb-n-estimators 300 --gb-learning-rate 0.05

# Sin generar graficos ni dashboard (solo entrenar y exportar)
python malnutrition_model.py --model gradient_boosting --export-model \
    --no-plots --no-dashboard
```

---

## 2. Dependencias

```bash
pip install scikit-learn joblib pandas numpy
```

---

## 3. Uso de `predict.py`

### Clase principal: `MalnutritionPredictor`

El archivo `predict.py` expone la clase `MalnutritionPredictor` que encapsula la carga del modelo, el preprocesamiento de datos y la prediccion.

### Inicializacion

```python
from predict import MalnutritionPredictor

# Cargar modelo desde el directorio de empaquetamiento
predictor = MalnutritionPredictor("output/model_package")
```

**Parametro**: ruta al directorio que contiene los artefactos del modelo (`model.joblib`, `imputer.joblib`, `pipeline_metadata.json`, `feature_names.json`).

---

### Metodo `predict(data)` - Realizar predicciones

Recibe datos crudos de paciente(s) y devuelve la prediccion con probabilidad.

#### Entrada

El parametro `data` acepta tres formatos:

| Formato | Tipo | Uso |
|---------|------|-----|
| Diccionario | `dict` | Un solo paciente |
| Lista de diccionarios | `list[dict]` | Multiples pacientes (batch) |
| DataFrame | `pd.DataFrame` | Multiples pacientes |

Cada diccionario contiene las variables clinicas del paciente como pares `clave: valor`. **No es necesario enviar todas las variables**; las faltantes se imputaran automaticamente con la mediana del dataset de entrenamiento.

#### Ejemplo - Prediccion individual

```python
resultado = predictor.predict({
    "CSP_EscolaridadPadre": 5,
    "ERN_Peso": 1850,
    "ERN_Talla": 43.5,
    "ERN_PC": 30.2,
    "ERN_Sexo": 1,
    "ERN_Ballard": 34,
    "CP_edadmaterna": 28,
    "CP_TallaMadre": 160,
    "CP_PesoMadre": 62,
    "CP_TallaPadre": 172,
    "CP_PesoPadre": 78,
    "CP_TotalCPN": 6,
    "CP_NumEcografias": 4,
    "HD_TotalDiasHospital": 12,
    "HD_DiasURN": 5,
    "HD_DiasUCI": 3,
    "HD_PesoSalida": 2100,
    "BMImadre": 24.2,
    "BMIpadre": 26.4,
    "zscorepeso0": -1.5,
    "zscoretalla0": -1.2,
    "zscorepesotalla0": -0.8,
    "zscorePC0": -0.5,
    "zscorepeso1": -1.0,
    "zscoretalla1": -0.9,
    "zscorePC1": -0.3,
    "BMI1": 13.5,
    "zscoreBMI1": -1.1,
})
```

#### Ejemplo - Prediccion batch (multiples pacientes)

```python
resultados = predictor.predict([
    {
        "CSP_EscolaridadPadre": 5,
        "ERN_Peso": 1850,
        "ERN_Ballard": 34,
        "CP_edadmaterna": 28,
        "HD_TotalDiasHospital": 12,
    },
    {
        "CSP_EscolaridadPadre": 2,
        "ERN_Peso": 980,
        "ERN_Ballard": 28,
        "CP_edadmaterna": 17,
        "HD_TotalDiasHospital": 45,
    },
])
```

#### Salida

El metodo `predict()` retorna un diccionario con la siguiente estructura:

```python
{
    "predictions": [1],          # Lista de predicciones (0 o 1)
    "probabilities": [0.7234],   # Lista de probabilidades de crecimiento armonico
    "labels": ["Crecimiento armónico (adecuado)"],  # Etiquetas legibles
    "risk_scores": [0.2766],     # Score de riesgo de malnutricion (1 - probability)
    "model_info": {
        "model_name": "Gradient Boosting",
        "metrics": {
            "accuracy": 0.7768,
            "balanced_accuracy": 0.7345,
            "f1_score": 0.835,
            "roc_auc": 0.813,
            "cv_f1_mean": 0.8302,
            "cv_f1_std": 0.0049
        },
        "n_features": 87
    }
}
```

**Detalle de cada campo de salida:**

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `predictions` | `list[int]` | `0` = Malnutricion, `1` = Crecimiento armonico. Un valor por paciente |
| `probabilities` | `list[float]` | Probabilidad de crecimiento armonico (0 a 1). Un valor por paciente |
| `labels` | `list[str]` | Etiqueta legible del resultado. Un valor por paciente |
| `risk_scores` | `list[float]` | Score de riesgo de malnutricion = `1 - probability`. Un valor por paciente |
| `model_info` | `dict` | Metadatos del modelo: nombre, metricas de evaluacion, numero de features |

Para un batch de N pacientes, cada lista tendra N elementos en el mismo orden de entrada.

---

### Metodo `get_feature_names()` - Consultar features del modelo

Retorna la lista completa de features que el modelo espera. Util para saber que variables se pueden enviar.

```python
features = predictor.get_feature_names()
print(len(features))  # 87
print(features[:5])   # ['CSP_EscolaridadPadre', 'V195B', 'zscoretalla1', ...]
```

---

### Metodo `get_model_info()` - Consultar metadatos del modelo

Retorna un diccionario con la informacion del modelo cargado.

```python
info = predictor.get_model_info()
```

**Salida:**

```python
{
    "model_name": "Gradient Boosting",
    "target": "indexnutricion12meses",
    "target_labels": {
        "0": "Malnutrición / Crecimiento NO armónico",
        "1": "Crecimiento armónico (adecuado)"
    },
    "metrics": {
        "accuracy": 0.7768,
        "balanced_accuracy": 0.7345,
        "f1_score": 0.835,
        "roc_auc": 0.813,
        "cv_f1_mean": 0.8302,
        "cv_f1_std": 0.0049
    },
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    "n_features": 87
}
```

---

### Uso como script standalone

Tambien se puede ejecutar `predict.py` directamente desde la terminal pasando un archivo JSON:

```bash
python predict.py --input datos_paciente.json --package-dir output/model_package
```

Donde `datos_paciente.json` puede ser un diccionario (un paciente) o una lista de diccionarios (batch):

```json
{
    "CSP_EscolaridadPadre": 5,
    "ERN_Peso": 1850,
    "ERN_Ballard": 34,
    "CP_edadmaterna": 28
}
```

La salida se imprime en formato JSON en stdout.

---

## 4. Variables de Entrada (Top 30 mas relevantes)

Las siguientes son las variables con mayor impacto en la prediccion. **No es necesario enviar todas**; las faltantes se imputaran con la mediana del dataset de entrenamiento.

| Variable | Importancia | Descripcion |
|----------|------------|-------------|
| `CSP_EscolaridadPadre` | 17.7% | Escolaridad del padre (nivel educativo) |
| `V195B` | 9.7% | Variable clinica prenatal |
| `zscoretalla1` | 8.7% | Z-score de talla a la entrada PMC |
| `BMI1` | 2.7% | Indice de masa corporal a la entrada PMC |
| `zscoretalla0` | 2.5% | Z-score de talla al nacimiento |
| `zscorePC1` | 2.1% | Z-score de perimetro cefalico a la entrada PMC |
| `CP_TallaMadre` | 2.1% | Talla de la madre (cm) |
| `zscoreBMI1` | 2.1% | Z-score de BMI a la entrada PMC |
| `zscorepeso0` | 1.9% | Z-score de peso al nacimiento |
| `zscorepesotalla0` | 1.9% | Z-score peso/talla al nacimiento |
| `BMImadre` | 1.8% | IMC de la madre |
| `CP_TallaPadre` | 1.8% | Talla del padre (cm) |
| `zscorepeso1` | 1.8% | Z-score de peso a la entrada PMC |
| `ERN_Peso` | 1.2% | Peso al nacer (gramos) |
| `HD_DiasURN` | 1.2% | Dias en Unidad de Recien Nacidos |
| `CP_PesoPadre` | 1.2% | Peso del padre (kg) |
| `HD_PesoSalida` | 1.1% | Peso al salir del hospital (gramos) |
| `CP_edadmaterna` | 1.0% | Edad de la madre (anos) |
| `ERN_Sexo` | 1.0% | Sexo del recien nacido (1=M, 2=F) |
| `HD_TotalDiasHospital` | 1.0% | Dias totales de hospitalizacion |
| `CP_TotalCPN` | 1.0% | Total controles prenatales |
| `HD_DiasUCI` | 0.8% | Dias en UCI neonatal |
| `CP_PesoMadre` | 0.8% | Peso de la madre (kg) |
| `CP_NumEcografias` | 0.8% | Numero de ecografias realizadas |
| `BMIpadre` | 1.4% | IMC del padre |
| `ERN_Ballard` | - | Edad gestacional Ballard (semanas) |

---

## 5. Interpretacion de la Respuesta

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `predictions[i]` | `int` | `0` = Malnutricion, `1` = Crecimiento armonico |
| `probabilities[i]` | `float` | Probabilidad de crecimiento armonico (0 a 1) |
| `risk_scores[i]` | `float` | Score de riesgo de malnutricion (= 1 - probability) |
| `labels[i]` | `str` | Etiqueta legible del resultado |

### Criterios de decision sugeridos

| Escenario | Condicion | Accion sugerida |
|-----------|-----------|-----------------|
| Bajo riesgo | `risk_score < 0.3` | Seguimiento estandar |
| Riesgo moderado | `0.3 <= risk_score < 0.6` | Monitoreo frecuente |
| Alto riesgo | `risk_score >= 0.6` | Intervencion nutricional prioritaria |

> **Nota:** Estos umbrales son sugeridos y deben ser validados por el equipo medico.

---

## 6. Ejemplo completo de integracion

```python
from predict import MalnutritionPredictor

# 1. Cargar modelo
predictor = MalnutritionPredictor("output/model_package")

# 2. Consultar informacion del modelo
info = predictor.get_model_info()
print(f"Modelo: {info['model_name']}, F1: {info['metrics']['f1_score']}")

# 3. Consultar features disponibles
features = predictor.get_feature_names()
print(f"El modelo acepta {len(features)} features")

# 4. Prediccion individual
resultado = predictor.predict({
    "CSP_EscolaridadPadre": 5,
    "ERN_Peso": 1850,
    "ERN_Ballard": 34,
    "CP_edadmaterna": 28,
    "HD_TotalDiasHospital": 12,
})

prediccion = resultado["predictions"][0]       # 0 o 1
probabilidad = resultado["probabilities"][0]   # float entre 0 y 1
riesgo = resultado["risk_scores"][0]           # float entre 0 y 1
etiqueta = resultado["labels"][0]              # str legible

print(f"Prediccion: {etiqueta}")
print(f"Probabilidad de crecimiento armonico: {probabilidad:.1%}")
print(f"Score de riesgo de malnutricion: {riesgo:.1%}")

# 5. Prediccion batch
resultados = predictor.predict([
    {"ERN_Peso": 1850, "ERN_Ballard": 34, "CP_edadmaterna": 28},
    {"ERN_Peso": 980, "ERN_Ballard": 28, "CP_edadmaterna": 17},
])

for i, label in enumerate(resultados["labels"]):
    print(f"Paciente {i+1}: {label} (riesgo: {resultados['risk_scores'][i]:.1%})")
```

---

## 7. Arquitectura del Empaquetamiento

```
malnutrition_model.py              # Script de entrenamiento (con --export-model)
        |
        v
output/model_package/               # Artefactos serializados
    model.joblib                    # GradientBoostingClassifier entrenado
    imputer.joblib                  # SimpleImputer(strategy='median') ajustado
    pipeline_metadata.json          # Config completa del pipeline de preprocesamiento
    feature_names.json              # Orden exacto de features del modelo
        |
        v
predict.py                          # Modulo de prediccion (MalnutritionPredictor)
        |                             - Carga artefactos
        |                             - Aplica preprocesamiento identico al entrenamiento
        |                             - Expone predict(), get_feature_names(), get_model_info()
        v
    Su API / aplicacion              # Integre predict.py en su servicio
```
