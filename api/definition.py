# File handling
import json
from typing import Any

# Data analysis packages
import numpy as np
import pandas as pd

# API control
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

# Package versioning
#from model import __version__ as model_version
from schemas import predict #, __version__

# Model prediction
from model.predict import make_prediction

# App settings
from config import settings


api_router = APIRouter()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=predict.PredictionResults, status_code=200)
async def predict(input_data: predict.MultipleDataInputs) -> Any:
    """
    Prediccion usando el modelo de PrematureBirth
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Haciendo prediccion sobre los siguientes inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Error en prediccion: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Resultados de prediccion: {results.get('predictions')}")

    return results
