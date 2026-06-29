# services/mortality_service.py

import json
from pathlib import Path

import joblib
import pandas as pd

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.mortality_prediction import MortalityPrediction
from config.paths import (MODELS_DIR, ARTIFACTS_DIR)
class MortalityService:

    FEATURE_COLUMNS = [
            "age",
            "bmi",
            "bmi_squared",
            "gender",
            "occupation_risk",
            "alcohol_use",
            "exercise_level",
            "smoker_status",
            "diabetes",
            "hypertension",
            "heart_disease",
            "respiratory_disease",
            "cancer_history",
        ]

    def __init__(
        self,
        model_path: Path | str = "mortality_prediction_model_v1.joblib",
        thresholds_path: Path | str = (
            ARTIFACTS_DIR
            / "mortality_model"
            / "risk_thresholds.json"
        ),
    ):

        self.model = joblib.load(
            MODELS_DIR / model_path
        )

        with open(thresholds_path, "r") as f:
            self.thresholds = json.load(f)

    # ==========================================
    # PUBLIC
    # ==========================================

    def predict(self, customer: EnrichedCustomerProfile,) -> MortalityPrediction:

        features = self._build_features(customer)

        probability = (self.model.predict_proba(features)[0][1])

        category = (self._get_category(probability))

        score = probability * 100

        return MortalityPrediction(
            mortality_probability=probability,
            mortality_category=category,
            mortality_score=round(score, 2),
        )

    # ==========================================
    # FEATURE BUILDING
    # ==========================================

    def _build_features(
        self,
        customer: EnrichedCustomerProfile,
    ) -> pd.DataFrame:

        row = {
            "age": customer.age,
            "bmi": customer.bmi,
            "bmi_squared": customer.bmi ** 2,
            "gender": customer.gender,
            "occupation_risk": customer.occupation_risk,
            "alcohol_use": customer.alcohol_use,
            "exercise_level": customer.exercise_level,
            "smoker_status": customer.smoker_status,
            "diabetes": customer.diabetes,
            "hypertension": customer.hypertension,
            "heart_disease": customer.heart_disease,
            "respiratory_disease": customer.respiratory_disease,
            "cancer_history": customer.cancer_history,
        }

        df = pd.DataFrame([row])

        return df[self.FEATURE_COLUMNS]


    # ==========================================
    # RISK BANDS
    # ==========================================

    def _get_category(
    self,
    probability: float,
    ) -> str:
        for category, bounds in self.thresholds.items():
            if (bounds["min"] <= probability < bounds["max"]):
                return category.capitalize()
        return "Severe"