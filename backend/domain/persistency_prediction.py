# domain/persistency_prediction.py

from dataclasses import dataclass


@dataclass
class PersistencyPrediction:

    product_id: str

    product_name: str

    lapse_probability: float

    lapse_category: str