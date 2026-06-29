# services/recommendation_service.py

from domain.recommendation_result import RecommendationResult
from domain.mortality_prediction import MortalityPrediction
from domain.coverage_recommendation import CoverageRecommendation
from domain.product_recommendation_result import ProductRecommendationResult
from domain.premium_estimate import PremiumEstimate
from domain.persistency_prediction import PersistencyPrediction
from domain.recommended_product import RecommendedProduct

class RecommendationService:

    def assemble(
        self,
        coverage_recommendation: CoverageRecommendation,
        product_recommendation: ProductRecommendationResult,
        premium_estimates: list[PremiumEstimate],
        persistency_predictions: list[PersistencyPrediction],
        mortality_prediction: MortalityPrediction,
    ) -> RecommendationResult:

        premium_lookup = {p.product_id: p for p in premium_estimates}
        persistency_lookup = {p.product_id: p for p in persistency_predictions}

        recommended_products = []
        for product in (product_recommendation.recommendations):
            premium = premium_lookup.get(product.product_id)
            persistency = persistency_lookup.get(product.product_id)
            if premium is None:
                continue
            if persistency is None:
                continue
            recommended_products.append(
                RecommendedProduct(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    product_type=product.product_type,
                    suitability_score=product.suitability_score,
                    annual_premium=premium.annual_premium,
                    monthly_premium=premium.monthly_premium,
                    lapse_probability=persistency.lapse_probability,
                    lapse_category=persistency.lapse_category,
                    rationale=product.rationale,
                )

            )

        recommended_products.sort(
            key=lambda x: (
                x.suitability_score,
                -x.lapse_probability,
            ),
            reverse=True,
        )

        return RecommendationResult(
            mortality_prediction=mortality_prediction,
            coverage_recommendation=coverage_recommendation,
            recommended_products=recommended_products,
        )