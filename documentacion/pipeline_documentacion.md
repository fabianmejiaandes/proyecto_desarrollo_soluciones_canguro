# Pipeline de Modelado — Predicción de Malnutrición a 12 meses EC
## PMCI / Fundación Canguro

**Fecha:** Abril 2026  
**Notebook de referencia:** `Modelado_Malnutricion-final.ipynb`

---

## 1. Contexto y Objetivo

El objetivo es predecir si un neonato prematuro presentará malnutrición a los 12 meses de edad corregida (EC), utilizando datos clínicos recolectados en diferentes momentos del seguimiento del Programa Madre Canguro Integral (PMCI) de la Fundación Canguro.

Se predicen tres outcomes de malnutrición definidos por z-scores OMS:

| Outcome | Definición | Variable |
|---------|-----------|----------|
| Stunting (talla baja) | HAZ < −2 | `zscoretalla12cat == 1` |
| Bajo peso | WAZ < −2 | `zscorepeso12cat == 1` |
| Wasting (emaciación) | WHZ < −2 | `zscorepesotalla12cat == 1` |

---

## 2. Fuente de Datos

- **Archivo:** `KMC-70k-93-2024-Malnutricion-conVel-DATA-SPSS-20250322.xlsx`
- **Total de registros:** 64,801
- **Total de variables:** 753
- **Periodo cubierto:** 2007–2023 (columna `Iden_FechaParto`)
- **Valores faltantes SPSS:** codificados como `#NULL!` → convertidos a `NaN`

### 2.1 Análisis de cobertura temporal

La columna `periodosanalisis` segmenta los datos en cohortes históricas:

| Periodo | Años | N registros | Disponibilidad fechas |
|---------|------|-------------|----------------------|
| P1–P3 | Sin fecha | ~12,640 | 0% — excluidos del análisis de cohortes |
| **P4** | **2007–2012** | **9,599** | **100%** |
| **P5** | **2013–2017** | **18,965** | **100%** |
| **P6** | **2018–2022** | **19,347** | **100%** |
| #NULL! | 2023 | 3,990 | 100% — seguimiento incompleto (~68% sin outcome) |

**Decisión:** El análisis principal usa los periodos P4, P5 y P6 (n=47,911). Los registros de 2023 se incluyen opcionalmente como análisis de sensibilidad, filtrando solo los que tienen outcome completo.

---

## 3. Preprocesamiento

### 3.1 Variables objetivo
Se construyeron variables binarias a partir de las categorías de z-score:
```
stunting12m      = 1 si zscoretalla12cat == 1, sino 0
underweight12m_b = 1 si zscorepeso12cat == 1, sino 0
wasting12m       = 1 si zscorepesotalla12cat == 1, sino 0
```

### 3.2 Control de leakage
Se identificaron y excluyeron variables derivadas de datos posteriores a los 9 meses o del propio outcome de 12 meses (e.g., `indexnutricion12meses`, `mortalidad40sem12meses`, `velocidad12_9mesesOMS`). También se excluyó cualquier variable con `'12'` en el nombre que no fuera el outcome.

### 3.3 Conversión de tipos
Todas las columnas con ≥50% de valores numéricos válidos fueron convertidas a tipo numérico. Las restantes se mantuvieron como categóricas.

---

## 4. Estrategia de Modelado: Cascada Temporal

Se diseñó una **cascada temporal de 7 fases**, donde cada fase acumula las variables disponibles hasta ese momento clínico:

| Fase | Momento clínico | Features acumuladas |
|------|----------------|---------------------|
| F0 | Prenatal / Parto | 41 |
| F1 | Nacimiento | 75 |
| F2 | Hospitalización | 107 |
| F3 | 40 semanas EC | 137 |
| F4 | 3 meses EC | 164 |
| F5 | 6 meses EC | 183 |
| F6 | 9 meses EC | 198 |

Esta estrategia permite evaluar **cuándo emerge la señal predictiva** y ofrece predicciones en cualquier punto del seguimiento clínico.

---

## 5. Modelo: LightGBM con Validación Cruzada

Se entrenó un modelo **LightGBM** para cada combinación de (fase × outcome), usando validación cruzada estratificada de 5 folds.

### Hiperparámetros principales

| Parámetro | Valor |
|-----------|-------|
| objective | binary |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_child_samples | 30 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| early_stopping | 50 rounds |
| scale_pos_weight | automático (n_neg/n_pos) |

El `scale_pos_weight` se calculó automáticamente para cada fold y outcome, compensando el desbalance de clases.

---

## 6. Resultados: AUC por Fase y Outcome

### ROC-AUC media (5-fold CV)

| Fase | Stunting | Bajo peso | Wasting |
|------|----------|-----------|---------|
| F0 Prenatal/Parto | 0.645 | 0.618 | 0.555 |
| F1 Nacimiento | 0.737 | 0.751 | 0.689 |
| F2 Hospitalización | 0.741 | 0.754 | 0.705 |
| F3 40 semanas | 0.768 | 0.773 | 0.725 |
| F4 3 meses | 0.821 | 0.874 | 0.826 |
| F5 6 meses | 0.894 | 0.936 | 0.895 |
| **F6 9 meses** | **0.929** | **0.963** | **0.925** |

**n por outcome:** ~30,000 registros con outcome válido en cada caso.

### Métricas en la mejor fase (F6_9meses)

| Outcome | AUC | Sensibilidad | Especificidad | F1 |
|---------|-----|-------------|---------------|----|
| Stunting | 0.929 | 0.834 | 0.859 | 0.737 |
| Bajo peso | 0.963 | 0.844 | 0.941 | 0.726 |
| Wasting | 0.925 | 0.684 | 0.949 | 0.477 |

---

## 7. Interpretabilidad: SHAP Values

Se calcularon valores SHAP para el outcome principal (Stunting) en la mejor fase (F6_9meses), usando una muestra de 3,000 pacientes.

### Top 10 factores de riesgo — Stunting (F6_9meses)

| # | Variable | SHAP medio |
|---|----------|-----------|
| 1 | zscoretalla9 | 1.3241 |
| 2 | zscoretalla6 | 0.6952 |
| 3 | zscorepeso9 | 0.1527 |
| 4 | velocidad9_6mesesOMS | 0.1122 |
| 5 | zscoretalla9cat | 0.0928 |
| 6 | zscorepeso6 | 0.0580 |
| 7 | zscoretalla2 | 0.0555 |
| 8 | zscoretalla6cat | 0.0552 |
| 9 | velocidad6_3mesesOMS | 0.0335 |
| 10 | CP_TallaMadre | 0.0286 |

Los z-scores de talla a los 9 y 6 meses son los predictores más importantes, seguidos de la velocidad de crecimiento. La talla de la madre aparece como factor de riesgo contextual relevante.

---

## 8. Sistema de Riesgo Dinámico

Se calculó la probabilidad de Stunting en cada fase para todos los pacientes con outcome conocido, permitiendo visualizar la **trayectoria de riesgo individual** a lo largo del seguimiento.

### Separación media entre grupos (stunted vs normal) por fase

| Fase | P(stunted) | P(normal) | Delta |
|------|-----------|-----------|-------|
| F0 Prenatal | 0.562 | 0.457 | 0.106 |
| F1 Nacimiento | 0.628 | 0.421 | 0.207 |
| F2 Hospitalización | 0.642 | 0.382 | 0.260 |
| F3 40 semanas | 0.703 | 0.338 | 0.365 |
| F4 3 meses | 0.742 | 0.289 | 0.454 |
| F5 6 meses | 0.767 | 0.183 | 0.584 |
| F6 9 meses | 0.787 | 0.145 | 0.641 |

La separación entre grupos aumenta progresivamente con cada fase, validando la utilidad clínica de la cascada temporal.

---

## 9. Comparación con Baseline

Se comparó LightGBM contra una Regresión Logística con regularización L1 (Lasso), usando las mismas features de la mejor fase (F6_9meses, Stunting):

| Modelo | AUC (5-fold CV) |
|--------|----------------|
| Regresión Logística L1 | 0.9216 ± 0.0031 |
| **LightGBM** | **0.9290** |
| Ganancia LightGBM | +0.0074 |

La regresión L1 seleccionó 153 de 198 variables. LightGBM supera al baseline aunque el margen es moderado en la fase final, lo que sugiere que las relaciones lineales explican gran parte de la señal en F6.

---

## 10. Análisis de Cohortes Temporales

Se comparó el modelo y las prevalencias entre los tres periodos históricos (P4, P5, P6).

### Prevalencia de Stunting por cohorte

| Cohorte | N | Prevalencia | IC 95% |
|---------|---|-------------|--------|
| P4: 2007–2012 | 4,646 | 27.1% | [25.9% – 28.4%] |
| P5: 2013–2017 | 9,890 | 21.4% | [20.6% – 22.2%] |
| P6: 2018–2022 | 9,226 | 19.5% | [18.7% – 20.3%] |

Se observa una **reducción sostenida de la prevalencia de Stunting** entre 2007 y 2022, lo que sugiere mejoras en los protocolos de atención del PMCI.

### Manejo de features nuevas entre cohortes
Se implementó la función `align_features_across_cohorts()` que detecta automáticamente features que aparecen en cohortes posteriores. LightGBM maneja estas features ausentes como `NaN` de forma nativa, sin necesidad de imputación manual.

### Validación cruzada entre cohortes
Se evaluó la generalización entrenando en un periodo y probando en otro (Stunting, F2_Hospitalización), verificando que el modelo mantiene capacidad predictiva al aplicarse a cohortes no vistas durante el entrenamiento.

### Análisis de sensibilidad (2007–2023)
Se comparó el AUC del análisis principal (P4–P6) contra la inclusión de registros 2023 con outcome completo. Si los AUC son similares, los resultados son robustos al criterio de inclusión temporal.

---

## 11. Glosario de Variables Relevantes

Las variables de z-score y velocidad de crecimiento son **variables derivadas**: se calculan a partir de las mediciones brutas (talla en cm, peso en gramos) aplicando los estándares de crecimiento de la OMS, por lo que no aparecen directamente en el diccionario de variables originales.

### Convención de nomenclatura

| Elemento | Significado |
|----------|-------------|
| `zscore` | Z-score: distancia en desviaciones estándar (DS) respecto a la mediana OMS para la edad y sexo del neonato |
| `talla` | Longitud/talla del neonato (cm) |
| `peso` | Peso del neonato (kg) |
| `...2` | Medición en la visita de **40 semanas** de edad corregida (EC) |
| `...6`, `...9` | Medición a los **6 o 9 meses** de EC respectivamente |
| `...cat` | Versión **categórica** del z-score (1 = desnutrición < −2 DS, 2 = normal, 3 = sobrepeso) |
| `velocidad` | **Velocidad de crecimiento** en talla entre dos visitas consecutivas (cm/mes), según estándares OMS |
| `CP_` | Variable registrada en la **Consulta Prenatal** |

### Descripción de las 10 variables más importantes — Stunting a 12 meses EC

| # | Variable | Descripción | Unidad | SHAP |
|---|----------|-------------|--------|------|
| 1 | `zscoretalla9` | Z-score de talla para la edad a los **9 meses EC**. Indica qué tan por debajo (negativo) o por encima (positivo) de la mediana OMS está la longitud del neonato. Es el predictor más potente: un z-score muy negativo a los 9 meses anticipa directamente el Stunting a los 12 meses. | DS | 1.3241 |
| 2 | `zscoretalla6` | Z-score de talla para la edad a los **6 meses EC**. Mismo indicador medido 3 meses antes. Confirma que la trayectoria de crecimiento en talla es el factor central del riesgo. | DS | 0.6952 |
| 3 | `zscorepeso9` | Z-score de peso para la edad a los **9 meses EC**. Un peso bajo acompaña frecuentemente la talla baja en neonatos prematuros con restricción de crecimiento. | DS | 0.1527 |
| 4 | `velocidad9_6mesesOMS` | **Velocidad de crecimiento en talla entre los 6 y 9 meses EC**, según estándares OMS. Captura si el neonato está acelerando o desacelerando su crecimiento. Una velocidad baja en esta ventana es señal de alerta crítica. | cm/mes | 0.1122 |
| 5 | `zscoretalla9cat` | **Versión categórica** del z-score de talla a los 9 meses EC. Aporta información adicional en los umbrales clínicos (< −2 DS = Stunting confirmado). | Categoría | 0.0928 |
| 6 | `zscorepeso6` | Z-score de peso para la edad a los **6 meses EC**. Complementa la información de talla con el estado nutricional en peso a mitad del seguimiento. | DS | 0.0580 |
| 7 | `zscoretalla2` | Z-score de talla en la visita de **40 semanas EC** (primera consulta ambulatoria post-egreso hospitalario). Un valor bajo en esta visita temprana ya anticipa riesgo a largo plazo. | DS | 0.0555 |
| 8 | `zscoretalla6cat` | **Versión categórica** del z-score de talla a los 6 meses EC. | Categoría | 0.0552 |
| 9 | `velocidad6_3mesesOMS` | **Velocidad de crecimiento en talla entre los 3 y 6 meses EC**, según OMS. Refleja el período de recuperación post-natal temprana. Una velocidad alta en esta ventana puede compensar déficits previos. | cm/mes | 0.0335 |
| 10 | `CP_TallaMadre` | **Talla de la madre (cm)**, registrada en la consulta prenatal. Factor genético y de contexto: madres de menor estatura tienden a tener hijos con menor potencial de crecimiento lineal. Es el único predictor no modificable del top 10. | cm | 0.0286 |

### Lectura clínica del modelo

Los 9 primeros predictores son **mediciones seriadas del propio neonato** (talla, peso y velocidad de crecimiento en distintos momentos del seguimiento). Esto confirma que el **patrón de crecimiento acumulado** es el mejor predictor del Stunting a los 12 meses EC.

> **Implicación clínica:** un neonato con z-score de talla negativo y baja velocidad de crecimiento entre los 6 y 9 meses EC tiene alta probabilidad de Stunting a los 12 meses. La ventana 6–9 meses EC es el momento crítico de intervención nutricional.

---

## 12. Modelos Guardados

Se entrenaron modelos finales sobre el **100% de los datos** (sin validación cruzada), usando el número de iteraciones promedio de los folds CV como `num_boost_round`.

**Estructura de archivos guardados en `modelos_pmci/`:**

```
modelos_pmci/
├── modelo_Stunting_F0_Prenatal_Parto.lgb
├── modelo_Stunting_F1_Nacimiento.lgb
├── ...
├── modelo_Wasting_F6_9meses.lgb      ← 21 modelos en total
├── features_F0_Prenatal_Parto.json   ← lista de features por fase
├── ...
├── features_F6_9meses.json
└── metadata.json                     ← AUC de referencia, fechas, rutas
```

**Total:** 21 modelos (7 fases × 3 outcomes) en formato nativo LightGBM (`.lgb`).

---

## 13. Inferencia sobre Paciente Nuevo

Se implementó la función `predecir_riesgo()` que permite predecir el riesgo de un paciente nuevo en cualquier fase:

```python
resultado = predecir_riesgo(
    paciente  = {'ERN_Peso': 1050, 'ERN_Talla': 35, ...},
    fase      = 'F1_Nacimiento',
    outcome   = 'Stunting',
    umbral    = 0.5
)
# Retorna: probabilidad, clasificación, features usadas/faltantes
```

**Características:**
- Acepta `dict`, `pd.Series` o `pd.DataFrame` de una fila
- Features faltantes → `NaN` (LightGBM las maneja nativamente)
- Features nuevas no vistas en entrenamiento → ignoradas automáticamente
- Reporta cuántas features se usaron vs. faltaron

---

## 14. Conclusiones

1. **LightGBM con cascada temporal** supera al baseline logístico y permite predicciones en cualquier punto del seguimiento clínico.
2. **Ya desde F1 (nacimiento)** existe capacidad predictiva útil (AUC ≥ 0.74 para Stunting), lo que permite alertas tempranas al egreso hospitalario.
3. **La señal predictiva aumenta con cada fase**, validando la arquitectura de cascada temporal.
4. **Los z-scores de talla a los 9 y 6 meses** son los predictores más importantes (SHAP), junto con la velocidad de crecimiento.
5. **La prevalencia de Stunting disminuyó** de 27.1% (2007–2012) a 19.5% (2018–2022), sugiriendo impacto positivo de los protocolos PMCI.
6. Los modelos son **robustos a features nuevas** que puedan aparecer en cohortes futuras.

---

## Archivos Generados

| Archivo | Contenido |
|---------|-----------|
| `Modelado_Malnutricion-final.ipynb` | Notebook completo ejecutado |
| `modelos_pmci/modelo_*.lgb` | 21 modelos LightGBM finales |
| `modelos_pmci/features_*.json` | Feature sets por fase |
| `modelos_pmci/metadata.json` | Metadata de modelos |
