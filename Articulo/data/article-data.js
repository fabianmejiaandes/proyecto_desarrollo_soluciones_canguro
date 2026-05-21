window.ARTICLE_DATA = {
  "project": {
    "title": "Detección temprana del riesgo de malnutrición a 12 meses en prematuros y recién nacidos con bajo peso",
    "records": "31.017 pacientes",
    "variables": "200 variables acumuladas máximas",
    "models": "28 modelos (4 desenlaces x 7 fases)",
    "population": "Niños del Programa Madre Canguro Integral de la Fundación Canguro",
    "generated": "2026-05-11 07:11",
    "method": "Validación cruzada estratificada de 5 particiones, con predicciones fuera de cada partición"
  },
  "phases": [
    {
      "id": "F0_Prenatal_Parto",
      "short": "F0",
      "label": "Prenatal y parto",
      "display": "F0 Prenatal y parto",
      "features": 41,
      "time": "Antecedentes maternos, embarazo y parto"
    },
    {
      "id": "F1_Nacimiento",
      "short": "F1",
      "label": "Nacimiento",
      "display": "F1 Nacimiento",
      "features": 77,
      "time": "Mediciones y condición inicial al nacer"
    },
    {
      "id": "F2_Hospitalizacion",
      "short": "F2",
      "label": "Hospitalización",
      "display": "F2 Hospitalización",
      "features": 109,
      "time": "Evolución intrahospitalaria y egreso"
    },
    {
      "id": "F3_40semanas",
      "short": "F3",
      "label": "40 semanas EC",
      "display": "F3 40 semanas EC",
      "features": 139,
      "time": "Primer punto de comparación corregida"
    },
    {
      "id": "F4_3meses",
      "short": "F4",
      "label": "3 meses EC",
      "display": "F4 3 meses EC",
      "features": 166,
      "time": "Crecimiento temprano ambulatorio"
    },
    {
      "id": "F5_6meses",
      "short": "F5",
      "label": "6 meses EC",
      "display": "F5 6 meses EC",
      "features": 185,
      "time": "Trayectoria intermedia de crecimiento"
    },
    {
      "id": "F6_9meses",
      "short": "F6",
      "label": "9 meses EC",
      "display": "F6 9 meses EC",
      "features": 200,
      "time": "Última lectura antes del desenlace a 12 meses"
    }
  ],
  "outcomes": [
    {
      "id": "Stunting",
      "label": "Talla para la edad (T/E)",
      "short": "T/E",
      "clinical": "Talla a 12 meses por debajo de −2 DE (OMS). En A1/A3 simétricos se exige además velocidad de talla < 1 cm/mes entre 6 y 12 meses.",
      "n": 30953,
      "positives": 5337,
      "prevalence": 17.2,
      "bestPhase": "F6 9 meses EC",
      "aucF0": 0.624,
      "aucF6": 0.929,
      "gain": 0.305,
      "sensF6": 0.824,
      "specF6": 0.878,
      "f1F6": 0.684
    },
    {
      "id": "Bajo_peso",
      "label": "Peso para la edad (P/E)",
      "short": "P/E",
      "clinical": "Peso a 12 meses por debajo de −2 DE (OMS). Aplica igual para los cuatro grupos A1–A4, sin regla adicional.",
      "n": 29897,
      "positives": 3239,
      "prevalence": 10.8,
      "bestPhase": "F6 9 meses EC",
      "aucF0": 0.618,
      "aucF6": 0.964,
      "gain": 0.346,
      "sensF6": 0.814,
      "specF6": 0.95,
      "f1F6": 0.731
    },
    {
      "id": "Wasting",
      "label": "Peso para la talla (P/T)",
      "short": "P/T",
      "clinical": "Peso para la talla a 12 meses por debajo de −2 DE (OMS). Refleja proporcionalidad actual; aplica igual en todos los grupos A1–A4.",
      "n": 29828,
      "positives": 1232,
      "prevalence": 4.1,
      "bestPhase": "F6 9 meses EC",
      "aucF0": 0.555,
      "aucF6": 0.926,
      "gain": 0.371,
      "sensF6": 0.68,
      "specF6": 0.95,
      "f1F6": 0.48
    },
    {
      "id": "Mixta",
      "label": "Mixta (P/T + T/E)",
      "short": "Mixta",
      "clinical": "Coexisten P/T y T/E por debajo de −2 DE al mismo tiempo. En A1/A3 simétricos se aplica primero la regla de velocidad de T/E.",
      "n": 29828,
      "positives": 326,
      "prevalence": 1.1,
      "bestPhase": "F6 9 meses EC",
      "aucF0": 0.546,
      "aucF6": 0.965,
      "gain": 0.419,
      "sensF6": 0.546,
      "specF6": 0.987,
      "f1F6": 0.397
    }
  ],
  "metricsMatrix": [
    {
      "phase": "F0 Prenatal y parto",
      "short": "F0",
      "Stunting": 0.624,
      "Bajo_peso": 0.618,
      "Wasting": 0.555,
      "Mixta": 0.546
    },
    {
      "phase": "F1 Nacimiento",
      "short": "F1",
      "Stunting": 0.748,
      "Bajo_peso": 0.752,
      "Wasting": 0.7,
      "Mixta": 0.724
    },
    {
      "phase": "F2 Hospitalización",
      "short": "F2",
      "Stunting": 0.751,
      "Bajo_peso": 0.758,
      "Wasting": 0.71,
      "Mixta": 0.746
    },
    {
      "phase": "F3 40 semanas EC",
      "short": "F3",
      "Stunting": 0.775,
      "Bajo_peso": 0.773,
      "Wasting": 0.731,
      "Mixta": 0.764
    },
    {
      "phase": "F4 3 meses EC",
      "short": "F4",
      "Stunting": 0.824,
      "Bajo_peso": 0.874,
      "Wasting": 0.822,
      "Mixta": 0.866
    },
    {
      "phase": "F5 6 meses EC",
      "short": "F5",
      "Stunting": 0.893,
      "Bajo_peso": 0.937,
      "Wasting": 0.894,
      "Mixta": 0.936
    },
    {
      "phase": "F6 9 meses EC",
      "short": "F6",
      "Stunting": 0.929,
      "Bajo_peso": 0.964,
      "Wasting": 0.926,
      "Mixta": 0.965
    }
  ],
  "topFeatures": [
    {
      "feature": "zscoretalla9",
      "label": "Puntaje Z de talla a los 9 meses de edad corregida",
      "importance": 1.142,
      "earliestPhase": "F6 9 meses EC"
    },
    {
      "feature": "zscoretalla6",
      "label": "Puntaje Z de talla a los 6 meses de edad corregida",
      "importance": 0.515,
      "earliestPhase": "F5 6 meses EC"
    },
    {
      "feature": "RCIUpesoytallanacer",
      "label": "Restricción de crecimiento intrauterino por peso y talla al nacer",
      "importance": 0.504,
      "earliestPhase": "F1 Nacimiento"
    },
    {
      "feature": "velocidad9_6mesesOMS",
      "label": "Velocidad de ganancia de peso de 6 a 9 meses (OMS)",
      "importance": 0.125,
      "earliestPhase": "F6 9 meses EC"
    },
    {
      "feature": "zscorepeso9",
      "label": "Puntaje Z de peso a los 9 meses de edad corregida",
      "importance": 0.124,
      "earliestPhase": "F6 9 meses EC"
    },
    {
      "feature": "zscoretalla9cat",
      "label": "Categoría de talla a los 9 meses de edad corregida",
      "importance": 0.072,
      "earliestPhase": "F6 9 meses EC"
    },
    {
      "feature": "simetrico",
      "label": "Restricción de crecimiento intrauterino simétrica",
      "importance": 0.071,
      "earliestPhase": "F1 Nacimiento"
    },
    {
      "feature": "zscoretalla2",
      "label": "Puntaje Z de talla a los 3 meses de edad corregida",
      "importance": 0.056,
      "earliestPhase": "F4 3 meses EC"
    },
    {
      "feature": "zscorepeso6",
      "label": "Puntaje Z de peso a los 6 meses de edad corregida",
      "importance": 0.048,
      "earliestPhase": "F5 6 meses EC"
    },
    {
      "feature": "zscoretalla0",
      "label": "Puntaje Z de talla al nacer",
      "importance": 0.045,
      "earliestPhase": "F1 Nacimiento"
    },
    {
      "feature": "velocidad6_3mesesOMS",
      "label": "Velocidad de ganancia de peso de 3 a 6 meses (OMS)",
      "importance": 0.041,
      "earliestPhase": "F5 6 meses EC"
    },
    {
      "feature": "zscorepeso0",
      "label": "Puntaje Z de peso al nacer",
      "importance": 0.036,
      "earliestPhase": "F1 Nacimiento"
    },
    {
      "feature": "zscorepesotalla9",
      "label": "Puntaje Z de peso para la talla a los 9 meses",
      "importance": 0.033,
      "earliestPhase": "F6 9 meses EC"
    },
    {
      "feature": "zscoretalla6cat",
      "label": "Categoría de talla a los 6 meses",
      "importance": 0.032,
      "earliestPhase": "F5 6 meses EC"
    },
    {
      "feature": "gananciapesonacerpesoentradaPMC",
      "label": "Ganancia de peso desde nacimiento hasta entrada al programa",
      "importance": 0.027,
      "earliestPhase": "F3 40 semanas EC"
    }
  ],
  "stageFeatureCharts": [
    {
      "outcome": "Stunting",
      "label": "Talla para la edad (T/E)",
      "short": "T/E",
      "image": "assets/charts/features_por_etapa_barras_stunting.svg"
    },
    {
      "outcome": "Bajo_peso",
      "label": "Peso para la edad (P/E)",
      "short": "P/E",
      "image": "assets/charts/features_por_etapa_barras_bajo_peso.svg"
    },
    {
      "outcome": "Wasting",
      "label": "Peso para la talla (P/T)",
      "short": "P/T",
      "image": "assets/charts/features_por_etapa_barras_wasting.svg"
    },
    {
      "outcome": "Mixta",
      "label": "Mixta (P/T + T/E)",
      "short": "Mixta",
      "image": "assets/charts/features_por_etapa_barras_mixta.svg"
    }
  ],
  "missingness": {
    "rows": 31017,
    "phases": [
      {
        "phase": "F0 Prenatal y parto",
        "short": "F0",
        "features": 41,
        "meanMissing": 22.3,
        "medianMissing": 15.5,
        "featuresOver50": 14.6,
        "medianPatientAvailable": 73.2,
        "patientsWith75Available": 42.1
      },
      {
        "phase": "F1 Nacimiento",
        "short": "F1",
        "features": 77,
        "meanMissing": 15.1,
        "medianMissing": 10.4,
        "featuresOver50": 7.8,
        "medianPatientAvailable": 84.4,
        "patientsWith75Available": 80.8
      },
      {
        "phase": "F2 Hospitalización",
        "short": "F2",
        "features": 109,
        "meanMissing": 19.2,
        "medianMissing": 15.4,
        "featuresOver50": 11.9,
        "medianPatientAvailable": 84.4,
        "patientsWith75Available": 79.4
      },
      {
        "phase": "F3 40 semanas EC",
        "short": "F3",
        "features": 139,
        "meanMissing": 19.4,
        "medianMissing": 15.4,
        "featuresOver50": 12.2,
        "medianPatientAvailable": 84.2,
        "patientsWith75Available": 77.9
      },
      {
        "phase": "F4 3 meses EC",
        "short": "F4",
        "features": 166,
        "meanMissing": 18.9,
        "medianMissing": 16.0,
        "featuresOver50": 10.2,
        "medianPatientAvailable": 84.9,
        "patientsWith75Available": 72.1
      },
      {
        "phase": "F5 6 meses EC",
        "short": "F5",
        "features": 185,
        "meanMissing": 18.5,
        "medianMissing": 15.5,
        "featuresOver50": 9.2,
        "medianPatientAvailable": 85.4,
        "patientsWith75Available": 73.2
      },
      {
        "phase": "F6 9 meses EC",
        "short": "F6",
        "features": 200,
        "meanMissing": 18.2,
        "medianMissing": 15.4,
        "featuresOver50": 8.5,
        "medianPatientAvailable": 85.0,
        "patientsWith75Available": 73.4
      }
    ],
    "topMissingF6": [
      {
        "feature": "HD_UltiValorHematocrito",
        "label": "Último valor de hematocrito",
        "missing": 82.8
      },
      {
        "feature": "surfactante",
        "label": "recibió surfactante y menos o igual a 32 semanas",
        "missing": 79.7
      },
      {
        "feature": "corticoprenatalmenos34",
        "label": "menos de 34 semanas y ciclos de corticoides prenatales",
        "missing": 71.9
      },
      {
        "feature": "HD_ValorMasAltoBilirubina",
        "label": "Valor más alto de bilirrubina",
        "missing": 70.6
      },
      {
        "feature": "tipoventilacion",
        "label": "tipo de ventilacion en UCI",
        "missing": 67.9
      },
      {
        "feature": "HD_NumTrasSanguineas",
        "label": "Número de transfusiones sanguíneas",
        "missing": 62.3
      },
      {
        "feature": "CSP_TipoVivienda",
        "label": "Tipo de vivienda",
        "missing": 61.3
      },
      {
        "feature": "PA_DiasHospiMadre",
        "label": "Días de hospitalización materna",
        "missing": 61.2
      }
    ]
  },
  "technical": {
    "trainingDate": "2026-05-11 07:12",
    "sourceNotebook": "Modelado_Malnutricion-final.ipynb",
    "metricMethod": "Validación cruzada estratificada de 5 particiones, con predicciones fuera de cada partición",
    "modelFamily": "LightGBM, árboles de decisión potenciados por gradiente",
    "roundRange": "50 a 130 iteraciones de boosting",
    "reportedModels": 28,
    "positiveCounts": {
      "Stunting": 5337,
      "Bajo_peso": 3239,
      "Wasting": 1232,
      "Mixta": 326
    },
    "phaseFeatureCounts": [
      {
        "phase": "F0 Prenatal y parto",
        "features": 41
      },
      {
        "phase": "F1 Nacimiento",
        "features": 77
      },
      {
        "phase": "F2 Hospitalización",
        "features": 109
      },
      {
        "phase": "F3 40 semanas EC",
        "features": 139
      },
      {
        "phase": "F4 3 meses EC",
        "features": 166
      },
      {
        "phase": "F5 6 meses EC",
        "features": 185
      },
      {
        "phase": "F6 9 meses EC",
        "features": 200
      }
    ],
    "outcomeBalance": [
      {
        "outcome": "Talla para la edad (T/E)",
        "nTotal": 30953,
        "nPositive": 5337,
        "prevalence": 17.2,
        "scalePosWeight": 4.8
      },
      {
        "outcome": "Peso para la edad (P/E)",
        "nTotal": 29897,
        "nPositive": 3239,
        "prevalence": 10.8,
        "scalePosWeight": 8.23
      },
      {
        "outcome": "Peso para la talla (P/T)",
        "nTotal": 29828,
        "nPositive": 1232,
        "prevalence": 4.1,
        "scalePosWeight": 23.21
      },
      {
        "outcome": "Mixta (P/T + T/E)",
        "nTotal": 29828,
        "nPositive": 326,
        "prevalence": 1.1,
        "scalePosWeight": 90.5
      }
    ],
    "params": [
      [
        "objective",
        "binary",
        "Cada modelo estima si aparece o no un desenlace."
      ],
      [
        "metric",
        "auc",
        "Optimiza y monitorea separación entre casos y no casos."
      ],
      [
        "learning_rate",
        "0.05",
        "Hace que cada árbol corrija de forma gradual al anterior."
      ],
      [
        "num_leaves",
        "63",
        "Permite capturar interacciones no lineales entre variables."
      ],
      [
        "min_child_samples",
        "30",
        "Evita reglas demasiado específicas para pocos pacientes."
      ],
      [
        "feature_fraction",
        "0.8",
        "Cada iteración usa una fracción de variables para reducir sobreajuste."
      ],
      [
        "bagging_fraction",
        "0.8",
        "Cada iteración usa una fracción de filas para estabilizar el entrenamiento."
      ],
      [
        "bagging_freq",
        "5",
        "Aplica bagging cada cinco iteraciones."
      ],
      [
        "reg_alpha / reg_lambda",
        "0.1 / 0.1",
        "Penaliza modelos excesivamente complejos."
      ],
      [
        "detención temprana (early_stopping)",
        "50",
        "Detiene el entrenamiento si el AUC de validación no mejora."
      ],
      [
        "seed",
        "42",
        "Hace reproducible la partición y el entrenamiento."
      ]
    ],
    "baseline": {
      "model": "Regresión logística L1 balanceada",
      "auc": 0.9104,
      "aucStd": 0.0051,
      "lgbmAuc": 0.9288,
      "gain": 0.0184,
      "selectedVariables": "161 de 200"
    },
    "artifacts": [
      [
        "modelos_pmci/modelo_<desenlace>_<fase>.lgb",
        "Modelo LightGBM entrenado por desenlace y fase."
      ],
      [
        "modelos_pmci/features_<fase>.json",
        "Lista de variables acumuladas disponibles en cada fase."
      ],
      [
        "metricas/metricas_por_fase.csv",
        "AUC, sensibilidad, especificidad y F1 por desenlace y fase."
      ],
      [
        "metricas/shap_importancia_global.csv",
        "Importancia SHAP global disponible para T/E en F6."
      ],
      [
        "metricas/shap_values.csv",
        "Valores SHAP por paciente para explicabilidad."
      ],
      [
        "metricas/pacientes_dashboard.csv",
        "Pacientes con desenlaces, predicciones por fase y variables para visualizaciones."
      ],
      [
        "metricas/cluster_perfiles.csv",
        "Perfiles agregados de trayectorias de riesgo."
      ]
    ]
  }
};
