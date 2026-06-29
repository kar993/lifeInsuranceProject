# services/persistency_service.py

from pathlib import Path

import joblib
import pandas as pd

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.coverage_recommendation import CoverageRecommendation
from domain.product_recommendation_result import ProductRecommendationResult
from domain.premium_estimate import PremiumEstimate
from domain.persistency_prediction import PersistencyPrediction
from config.paths import MODELS_DIR


class PersistencyService:

    FEATURE_COLUMNS = [

        "age",
        "annual_income",
        "dependents",
        "coverage_amount",
        "premium_amount",
        "policy_term",
        "premium_income_ratio",
        "coverage_income_ratio",
        "coverage_adequacy_ratio",
        "years_since_issue",

        "product_type",
        "payment_mode",
        "payment_mode_preference",

        "autopay_enabled",
        "autopay_preference",
        "payment_mode_matches_preference",
        "autopay_matches_preference",
    ]

    def __init__(
        self,
        model_path: str = "persistency_prediction_model_v1.joblib",
    ):
        self.model = joblib.load(MODELS_DIR / model_path)

    # ==========================================
    # PUBLIC
    # ==========================================

    def predict(
        self,
        customer: EnrichedCustomerProfile,
        coverage_result: CoverageRecommendation,
        product_result: ProductRecommendationResult,
        premium_estimates: list[PremiumEstimate],
    ) -> list[PersistencyPrediction]:

        predictions = []

        premium_lookup = {p.product_id: p for p in premium_estimates}

        for product in (product_result.recommendations):

            premium = premium_lookup[product.product_id]

            features = (
                self._build_features(
                    customer,
                    coverage_result,
                    product,
                    premium,
                )
            )

            lapse_probability = (self.model.predict_proba(features)[0][1])

            predictions.append(
                PersistencyPrediction(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    lapse_probability=lapse_probability,
                    lapse_category=self._get_category(lapse_probability),
                )

            )

        return predictions

    # ==========================================
    # FEATURE BUILDING
    # ==========================================

    def _build_features(
        self,
        customer,
        coverage_result,
        product,
        premium,
    ) -> pd.DataFrame:

        coverage_amount = (coverage_result.recommended_coverage)
        premium_amount = (premium.annual_premium)
        annual_income = (customer.annual_income)
        premium_income_ratio = (premium_amount / annual_income)
        policy_term = self._recommend_policy_term(customer.age)
        coverage_income_ratio = (coverage_amount / annual_income)
        coverage_adequacy_ratio = (coverage_amount / max(coverage_result.total_need, 1))
        payment_mode = (customer.payment_mode_preference)
        autopay_enabled = int(customer.autopay_preference)
        payment_mode_matches = 1
        autopay_matches = 1
        row = {
            "age":customer.age,
            "annual_income":annual_income,
            "dependents":customer.dependents,
            "coverage_amount":coverage_amount,
            "premium_amount":premium_amount,
            "policy_term":policy_term,
            "premium_income_ratio":premium_income_ratio,
            "coverage_income_ratio":coverage_income_ratio,
            "coverage_adequacy_ratio":coverage_adequacy_ratio,
            "years_since_issue":0,
            "product_type":product.product_type,
            "payment_mode":payment_mode,
            "payment_mode_preference":customer.payment_mode_preference,
            "autopay_enabled":autopay_enabled,
            "autopay_preference":int(customer.autopay_preference),
            "payment_mode_matches_preference":payment_mode_matches,
            "autopay_matches_preference":autopay_matches,
        }

        df = pd.DataFrame([row])

        return df[
            self.FEATURE_COLUMNS
        ]

    # ==========================================
    # RISK BANDS
    # ==========================================

    def _get_category(
        self,
        probability: float,
    ) -> str:

        if probability < 0.10:
            return "Low"

        if probability < 0.25:
            return "Medium"

        if probability < 0.50:
            return "High"

        return "Severe"

    # ======================================
    # Helper function
    # ======================================

    def _recommend_policy_term(self, age: int,) -> int:
        retirement_age = 65
        return max(5, retirement_age - age)