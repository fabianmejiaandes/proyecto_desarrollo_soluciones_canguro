# Artefactos en `output/`

Esta carpeta contiene los resultados generados por `malnutrition_model.py` (EDA, evaluación y comparación de modelos).

## Archivos principales

- **Comparación de modelos**
  - `model_comparison.csv`: tabla comparativa (Accuracy, Balanced Accuracy, F1, ROC-AUC, CV F1 mean/std).

- **Figuras de resultados**
  - `results_01_confusion_matrices.png`: matrices de confusión por modelo.
  - `results_02_roc_curves.png`: curvas ROC por modelo.
  - `results_03_precision_recall.png`: curvas Precision-Recall.

- **Importancia de variables (por modelo)**
  - `feature_importance_gradient_boosting.csv`
  - `feature_importance_random_forest.csv`
  - `results_04_feature_importance_gradient_boosting.png`
  - `results_04_feature_importance_random_forest.png`

## Componentes RF (Random Forest)

Los artefactos específicos de **Random Forest (RF)** son:

- **Importancia de variables**:
  - `feature_importance_random_forest.csv` (todas las variables con su importancia)
  - `results_04_feature_importance_random_forest.png` (top variables en gráfico)
- **Evaluación**:
  - Los resultados comparativos (incluyendo RF) quedan en `model_comparison.csv`
  - La matriz de confusión de RF queda dentro de `results_01_confusion_matrices.png`

Para regenerar estos artefactos:

```bash
python malnutrition_model.py --model random_forest
```

o para comparar todos:

```bash
python malnutrition_model.py --model all
```

## Modelo empaquetado para inferencia

- `model_package/`: carpeta generada cuando se usa `--export-model` (modelo + preprocesamiento).
  - `model.joblib`, `imputer.joblib`, `pipeline_metadata.json`, `feature_names.json`

