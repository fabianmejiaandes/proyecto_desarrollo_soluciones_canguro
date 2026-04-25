# Handover — Predicción de Malnutrición a 12 meses EC
## PMCI / Fundación Canguro

**Proyecto:** Identificación de factores de riesgo de malnutrición en niños prematuros evaluados a 12 meses de edad corregida  
**Fecha de corte:** Abril 2026  
**Contacto expertos:** Prof. José Tiberio Hernández · Nathalie Charpak MD · Catalina Lince MD

---

## 1. Contexto

Se busca predecir si un neonato prematuro presentará malnutrición a los **12 meses de edad corregida (EC)** usando datos clínicos longitudinales del Programa Madre Canguro Integral (PMCI). La base de datos contiene ~70.000 historias clínicas de los últimos 25 años.

**Tres outcomes de malnutrición (OMS):**

| Outcome | Criterio | Variable en BdD |
|---------|----------|-----------------|
| Stunting (talla baja) | HAZ < −2 DS | `zscoretalla12cat == 1` |
| Bajo peso | WAZ < −2 DS | `zscorepeso12cat == 1` |
| Wasting (desnutrición aguda) | WHZ < −2 DS | `zscorepesotalla12cat == 1` |

**Clasificación compuesta (4 grupos) — pendiente validación con neonatólogos:**

| Grupo | Criterio |
|-------|----------|
| 0 — Normal | Ningún indicador < −2 DS |
| 1 — Sobrepeso/Obesidad | `Overweightorobesity12m == 1`, sin desnutrición |
| 2 — Un déficit | Exactamente 1 de HAZ/WAZ/WHZ < −2 DS |
| 3 — Desnutrición múltiple | 2 o 3 indicadores < −2 DS |

---

## 2. Estructura de archivos del proyecto

```
Trabajo de grado/
│
├── DATOS
│   ├── KMC-70k-93-2024-Malnutricion-conVel-DATA-SPSS-20250322.xlsx  ← dataset principal
│   ├── KMC-70K-diccionarioVARS-Malnutricion-PhETI-rev20250520-MAIA.xlsx  ← diccionario de variables
│   └── feature_plan.json       ← asignación de variables a fases F0-F6
│
├── NOTEBOOKS (orden de ejecución)
│   ├── EDA_Malnutricion.ipynb          ← 1. Análisis exploratorio
│   ├── EDA_Prematurez.ipynb            ← 2. EDA específico de prematurez
│   ├── Mejoras_Modelo.ipynb            ← 3. Experimentos de mejora
│   └── Modelado_Malnutricion-1.ipynb   ← 4. Pipeline principal ⭐
│
├── MODELOS
│   └── modelos_pmci/
│       ├── modelo_{outcome}_{fase}.lgb  ← 21 modelos LightGBM
│       ├── features_{fase}.json         ← features por fase
│       └── metadata.json                ← AUC, fechas, rutas
│
├── DASHBOARD DATA
│   └── dashboard_data/
│       ├── README.json                  ← documentación para el dashboard
│       ├── test_predictions.csv         ← predicciones test set por paciente
│       ├── metricas_por_fase.csv        ← AUC/Sens/Spec por fase y outcome
│       ├── shap_importancia_global.csv  ← ranking de factores de riesgo
│       ├── shap_values.csv              ← SHAP individual por paciente
│       └── cohort_stats.csv             ← prevalencias por cohorte temporal
│
└── DOCUMENTACIÓN
    ├── HANDOVER.md                      ← este archivo
    ├── pipeline_documentacion.md/.pdf   ← documento técnico del pipeline
    └── Propuesta_Proyecto 1_Fundación Canguro (1).pdf  ← propuesta original
```

---

## 3. Notebook principal: `Modelado_Malnutricion-1.ipynb`

Es el notebook de referencia. Contiene todo el pipeline de principio a fin. **Ejecutar siempre de arriba a abajo.**

### Secciones

| Sección | Contenido |
|---------|-----------|
| **Setup** | Imports, carga del Excel, configuración de rutas |
| **1. Preprocesamiento** | Construcción de outcomes binarios, exclusión de variables con leakage, feature sets por fase |
| **1b. Clasificación compuesta** | Construcción del outcome de 4 grupos + visualización por cohorte |
| **2. Funciones** | `get_model_data()`, `train_lgbm_cv()` |
| **3. Cascada temporal** | Entrenamiento LightGBM 5-fold CV para cada fase × outcome |
| **4. Comparación AUC** | Tabla y gráficas de AUC por fase y outcome |
| **5. SHAP** | Interpretabilidad: top 20 factores de riesgo |
| **6. Riesgo dinámico** | Trayectorias de probabilidad por paciente a través de fases |
| **7. Baseline** | Comparación con Regresión Logística L1 |
| **8. Resumen** | Impresión de métricas finales |
| **9. Cohortes temporales** | Análisis por periodos P4/P5/P6 + validación cruzada entre cohortes |
| **10. Outcome compuesto** | Modelado multiclase (4 grupos) con LightGBM |
| **11. Train/Test + Exportación** | Split 80/20, entrenamiento final, generación de archivos para dashboard |
| **12. Inferencia** | Función `predecir_riesgo()` para predicción sobre paciente nuevo |

---

## 4. Cómo ejecutar

### Requisitos
```bash
pip install pandas numpy lightgbm shap scikit-learn matplotlib seaborn openpyxl pyarrow
```

### Ejecución completa
1. Abrir `Modelado_Malnutricion-1.ipynb` en Jupyter
2. Ejecutar la celda `%matplotlib inline` al inicio
3. **Kernel → Restart & Run All**
4. Al terminar (~30-45 min) se generan:
   - `modelos_pmci/` — 21 modelos entrenados
   - `dashboard_data/` — 6 archivos CSV/JSON para el dashboard

### Ejecución parcial (solo inferencia)
Si los modelos ya están guardados en `modelos_pmci/`, solo necesitas ejecutar:
- Celda de imports (Setup)
- Celda de preprocesamiento (para tener `cumulative_features`)
- Sección 12 (Inferencia)

---

## 5. Estrategia de modelado

### Cascada temporal
Se entrenó un modelo independiente para cada combinación de **(fase × outcome)**:
- **7 fases** (F0→F6): cada fase acumula las variables disponibles hasta ese momento clínico
- **3 outcomes binarios**: Stunting, Bajo peso, Wasting
- **1 outcome multiclase**: estado nutricional compuesto (4 grupos)
- **Total modelos guardados:** 21 (binarios) + 1 multiclase por fase = 28 modelos

### Modelo
- **Algoritmo:** LightGBM
- **Validación:** Stratified K-Fold, 5 folds
- **Desbalance:** `scale_pos_weight = n_neg / n_pos` (calculado automáticamente por fold)
- **Split train/test:** 80/20 aleatorio estratificado (`random_state=42`)

### Resultados clave (AUC en CV 5-fold, dataset completo)

| Fase | Stunting | Bajo peso | Wasting |
|------|----------|-----------|---------|
| F0 Prenatal | 0.645 | 0.618 | 0.555 |
| F1 Nacimiento | 0.737 | 0.751 | 0.689 |
| F2 Hospitalización | 0.741 | 0.754 | 0.705 |
| F3 40 semanas | 0.768 | 0.773 | 0.725 |
| F4 3 meses | 0.821 | 0.874 | 0.826 |
| F5 6 meses | 0.894 | 0.936 | 0.895 |
| **F6 9 meses** | **0.929** | **0.963** | **0.925** |

---

## 6. Datos — decisiones importantes

### Cobertura temporal
- La columna `periodosanalisis` (valores 4.0, 5.0, 6.0) define los cohortes históricos
- **P4: 2007–2012** (n≈9,600) | **P5: 2013–2017** (n≈18,965) | **P6: 2018–2022** (n≈19,347)
- Registros con `periodosanalisis` 1, 2, 3: sin fecha de parto disponible → excluidos del análisis temporal
- Registros 2023 (`periodosanalisis = NaN`): solo ~31% tiene outcome completo → usados en análisis de sensibilidad

### Leakage
Se excluyeron explícitamente variables con información del futuro (post 9 meses):
- Cualquier columna con `'12'` en el nombre (excepto los outcomes)
- `indexnutricion12meses`, `mortalidad40sem12meses`, `velocidad12_9mesesOMS`, etc.

### Valores faltantes
- Originalmente codificados como `'#NULL!'` en el Excel → convertidos a `NaN`
- LightGBM maneja NaN nativamente; **no se imputa**
- Variables con < 5% de disponibilidad en un cohorte se excluyen para ese cohorte

### Features nuevas entre cohortes
La función `align_features_across_cohorts()` detecta features que aparecen en cohortes posteriores. Se manejan con `reindex(fill_value=NaN)` — LightGBM las procesa sin error.

---

## 7. Cómo hacer inferencia sobre un paciente nuevo

```python
import lightgbm as lgb
import json
import pandas as pd
import numpy as np

def predecir_riesgo(paciente, fase, outcome='Stunting',
                    models_dir='modelos_pmci', umbral=0.5):
    """
    paciente : dict con variables clínicas del paciente
    fase     : 'F0_Prenatal_Parto' ... 'F6_9meses'
    outcome  : 'Stunting' | 'Bajo_peso' | 'Wasting'
    """
    safe_fase  = fase.replace('/', '_')
    model      = lgb.Booster(model_file=f'{models_dir}/modelo_{outcome}_{safe_fase}.lgb')
    features   = json.load(open(f'{models_dir}/features_{safe_fase}.json'))['features']

    X = pd.DataFrame([paciente]).reindex(columns=features, fill_value=np.nan)
    prob = float(model.predict(X)[0])
    return {'probabilidad': prob, 'riesgo': prob >= umbral}


# Ejemplo
paciente = {
    'ERN_Peso': 1050,          # peso al nacer (gramos)
    'ERN_Talla': 35,           # talla al nacer (cm)
    'ERN_EdadGestacional': 28, # semanas de gestación
    'CP_TallaMadre': 158,      # talla de la madre (cm)
}
resultado = predecir_riesgo(paciente, fase='F1_Nacimiento', outcome='Stunting')
print(f"P(Stunting) = {resultado['probabilidad']:.1%}")
```

**Importante:** features ausentes en el dict se completan con `NaN` automáticamente. Features nuevas no vistas durante el entrenamiento se ignoran.

---

## 8. Archivos para el dashboard

El equipo del dashboard lee los archivos de `dashboard_data/`. Ver `dashboard_data/README.json` para documentación completa de cada columna.

### Lectura rápida
```python
import pandas as pd, json

predicciones = pd.read_csv('dashboard_data/test_predictions.csv')
metricas     = pd.read_csv('dashboard_data/metricas_por_fase.csv')
shap_global  = pd.read_csv('dashboard_data/shap_importancia_global.csv')
shap_ind     = pd.read_csv('dashboard_data/shap_values.csv')
cohortes     = pd.read_csv('dashboard_data/cohort_stats.csv')
```

### Columnas clave de `test_predictions.csv`
- `prob_{outcome}_{fase}` — probabilidad de riesgo (0.0 a 1.0) para cada outcome × fase
- `real_{outcome}` — outcome real del paciente (0/1/NaN)
- `real_estado_nutricional` — grupo nutricional real (0/1/2/3)
- Variables clínicas de contexto: `ERN_Peso`, `ERN_Talla`, `cohort`, `Iden_Sede`, etc.

---

## 9. Pendiente / Próximos pasos

| Tarea | Responsable | Prioridad |
|-------|-------------|-----------|
| Validar clasificación de 4 grupos con neonatólogos (Charpak / Lince) | Investigador | Alta |
| Alinear periodos temporales con los exactos del documento (2005-2010, etc.) o justificar los actuales | Investigador | Media |
| Construir plataforma interactiva consumiendo `dashboard_data/` | Equipo dashboard | Alta |
| 1-2 casos de uso evaluados por equipo de neonatólogos | Neonatólogos | Alta |
| Revisión bibliográfica para fundamentar criterios de clasificación | Investigador | Media |

---

## 10. Variables más importantes (SHAP — Stunting, F6_9meses)

| # | Variable | Descripción | SHAP |
|---|----------|-------------|------|
| 1 | `zscoretalla9` | Z-score talla a los 9 meses EC | 1.3241 |
| 2 | `zscoretalla6` | Z-score talla a los 6 meses EC | 0.6952 |
| 3 | `zscorepeso9` | Z-score peso a los 9 meses EC | 0.1527 |
| 4 | `velocidad9_6mesesOMS` | Velocidad de crecimiento 6→9m EC (OMS) | 0.1122 |
| 5 | `zscoretalla9cat` | Categoría z-score talla 9m (discreta) | 0.0928 |
| 6 | `zscorepeso6` | Z-score peso a los 6 meses EC | 0.0580 |
| 7 | `zscoretalla2` | Z-score talla visita 40 semanas EC | 0.0555 |
| 8 | `zscoretalla6cat` | Categoría z-score talla 6m (discreta) | 0.0552 |
| 9 | `velocidad6_3mesesOMS` | Velocidad de crecimiento 3→6m EC (OMS) | 0.0335 |
| 10 | `CP_TallaMadre` | Talla de la madre (cm) — factor genético | 0.0286 |

**Interpretación:** los z-scores de talla a los 9 y 6 meses dominan la predicción. La ventana **6–9 meses EC** es el momento crítico de intervención nutricional.

---

## 11. Notas técnicas

- **Python:** 3.13 (miniforge)
- **LightGBM:** formato nativo `.lgb` (no pickle)
- **Datos:** el Excel tiene 64,801 filas × 753 columnas; carga ~2 min
- **Entrenamiento completo:** ~30-45 min (21 modelos × 5 folds)
- **`periodosanalisis`** viene como `float` en pandas (4.0, 5.0, 6.0) — siempre normalizar con `.str.replace(r'\.0$', '', regex=True)` antes de comparar
- **Backend matplotlib:** usar `%matplotlib inline` al inicio del notebook — NO usar `matplotlib.use('Agg')` o los plots no se muestran
- **`#NULL!`** en el Excel son valores faltantes SPSS — se reemplazan con `np.nan` en la carga
