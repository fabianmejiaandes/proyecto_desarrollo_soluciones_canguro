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