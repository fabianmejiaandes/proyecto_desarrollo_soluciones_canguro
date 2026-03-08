"""
Módulo de predicción para el modelo de malnutrición - Fundación Canguro
========================================================================
Carga el modelo empaquetado (Gradient Boosting) y realiza predicciones
sobre datos nuevos de pacientes.

Uso standalone:
    python predict.py --input datos_paciente.json
    python predict.py --input datos_multiples.json --package-dir output/model_package

Uso como módulo:
    from predict import MalnutritionPredictor
    predictor = MalnutritionPredictor("output/model_package")
    resultado = predictor.predict(datos_paciente)
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MalnutritionPredictor:
    """Predictor empaquetado para el modelo de malnutrición infantil.

    Carga el modelo entrenado y los artefactos del pipeline de preprocesamiento,
    y expone un método `predict()` que recibe datos crudos de un paciente
    y devuelve la predicción con probabilidad.
    """

    def __init__(self, package_dir: str = "output/model_package"):
        package_dir = Path(package_dir)

        self.model = joblib.load(package_dir / "model.joblib")
        self.imputer = joblib.load(package_dir / "imputer.joblib")

        with open(package_dir / "pipeline_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(package_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        self.missing_map = self.metadata["preprocessing"]["missing_map"]
        self.cat_to_onehot = self.metadata["feature_engineering"]["cat_to_onehot"]
        self.cat_to_drop = self.metadata["feature_engineering"]["cat_to_drop"]
        self.imputer_feature_names = self.metadata["feature_engineering"]["imputer_feature_names"]
        self.high_corr_features = self.metadata["feature_engineering"]["high_corr_features"]
        self.high_missing_cols = self.metadata["preprocessing"]["high_missing_cols"]
        self.zero_var_cols = self.metadata["preprocessing"]["zero_var_cols"]
        self.date_cols = self.metadata["preprocessing"]["date_cols"]

        logger.info(
            f"Modelo cargado: {self.metadata['model_name']} "
            f"({len(self.feature_names)} features)"
        )

    def _preprocess_input(self, data: dict | list[dict]) -> pd.DataFrame:
        """Aplica el mismo preprocesamiento usado en entrenamiento."""

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        # Convertir #NULL! a NaN
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace("#NULL!", np.nan)

        # Reemplazar missing values codificados
        for col in df.columns:
            if col in self.missing_map:
                df[col] = df[col].replace(float(self.missing_map[col]), np.nan)
            df[col] = df[col].replace(-1, np.nan)
            df[col] = df[col].replace(-1.0, np.nan)

        # Convertir columnas object a numéricas
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

        # Eliminar fechas
        date_cols = [c for c in self.date_cols if c in df.columns]
        df = df.drop(columns=date_cols, errors="ignore")

        # Eliminar columnas de alto missing / varianza cero
        df = df.drop(columns=[c for c in self.high_missing_cols if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in self.zero_var_cols if c in df.columns], errors="ignore")

        # One-hot encoding para categóricas
        cat_cols = [c for c in self.cat_to_onehot if c in df.columns]
        drop_cols = [c for c in self.cat_to_drop if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dummy_na=False)

        # Alinear con las features que espera el imputer (pre-colinealidad)
        df = df.reindex(columns=self.imputer_feature_names, fill_value=0)

        # Imputar valores faltantes con el imputer entrenado
        df = pd.DataFrame(
            self.imputer.transform(df), columns=self.imputer_feature_names, index=df.index
        )

        # Eliminar features de alta colinealidad
        cols_to_drop = [c for c in self.high_corr_features if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        return df

    def predict(self, data: dict | list[dict]) -> dict:
        """Realiza la predicción sobre datos de paciente(s).

        Args:
            data: Diccionario con los datos de un paciente, o lista de diccionarios
                  para múltiples pacientes.

        Returns:
            dict con:
                - predictions: lista de predicciones (0 o 1)
                - probabilities: lista de probabilidades de crecimiento armónico
                - labels: lista de etiquetas legibles
                - risk_scores: lista de scores de riesgo de malnutrición (1 - prob)
                - model_info: información del modelo usado
        """
        X = self._preprocess_input(data)

        predictions = self.model.predict(X).tolist()
        probabilities = self.model.predict_proba(X)[:, 1].tolist()

        labels = []
        risk_scores = []
        for pred, prob in zip(predictions, probabilities):
            if pred == 1:
                labels.append("Crecimiento armónico (adecuado)")
            else:
                labels.append("Malnutrición / Crecimiento NO armónico")
            risk_scores.append(round(1 - prob, 4))

        return {
            "predictions": predictions,
            "probabilities": [round(p, 4) for p in probabilities],
            "labels": labels,
            "risk_scores": risk_scores,
            "model_info": {
                "model_name": self.metadata["model_name"],
                "metrics": self.metadata["metrics"],
                "n_features": len(self.feature_names),
            },
        }

    def get_feature_names(self) -> list[str]:
        """Retorna la lista de features que el modelo espera."""
        return self.feature_names.copy()

    def get_model_info(self) -> dict:
        """Retorna la metadata del modelo."""
        return {
            "model_name": self.metadata["model_name"],
            "target": self.metadata["target"],
            "target_labels": self.metadata["target_labels"],
            "metrics": self.metadata["metrics"],
            "hyperparameters": self.metadata["hyperparameters"],
            "n_features": len(self.feature_names),
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Predicción de malnutrición infantil")
    parser.add_argument(
        "--input", required=True, help="Archivo JSON con datos del paciente"
    )
    parser.add_argument(
        "--package-dir",
        default="output/model_package",
        help="Directorio del modelo empaquetado",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    predictor = MalnutritionPredictor(args.package_dir)
    result = predictor.predict(patient_data)

    print(json.dumps(result, indent=2, ensure_ascii=False))
