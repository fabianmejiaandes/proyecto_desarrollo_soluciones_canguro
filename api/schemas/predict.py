from typing import Any, List, Optional

from pydantic import BaseModel


class DataInputSchema(BaseModel):
    pass

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'Feature': 1,
                    }
                ]
            }
        }
