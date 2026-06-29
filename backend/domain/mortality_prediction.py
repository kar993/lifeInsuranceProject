# domain/mortality_prediction.py

from dataclasses import dataclass


@dataclass
class MortalityPrediction:

    mortality_probability: float

    mortality_category: str

    mortality_score: float