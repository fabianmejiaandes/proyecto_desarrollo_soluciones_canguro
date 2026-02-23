import typing as t

import numpy as np
import pandas as pd

from model.pipeline import Pipeline

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)

    predictions = Pipeline.predict(
        X=data
    )
    results = {
        "predictions": [pred for pred in predictions],
    }

    return results