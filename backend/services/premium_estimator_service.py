# services/premium_estimator.py

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.coverage_recommendation import CoverageRecommendation
from domain.product_recommendation_result import ProductRecommendationResult
from domain.premium_estimate import PremiumEstimate
from domain.mortality_prediction import MortalityPrediction

class PremiumEstimator:
    """
    Estimates premiums for recommended products.

    Formula:

        base_premium =
            (coverage / 1000)
            * base_premium_rate

        final_premium =
            base_premium
            * age_factor
            * (1 + medical_loading)
    """

    def estimate(
        self,
        customer: EnrichedCustomerProfile,
        coverage_result: CoverageRecommendation,
        product_result: ProductRecommendationResult,
        mortality_event: MortalityPrediction,
    ) -> list[PremiumEstimate]:

        premium_estimates = []

        coverage_amount = coverage_result.recommended_coverage

        age_factor = self._get_age_factor(customer.age)
        
        medical_factor = (1 + customer.medical_extra_loading)

        mortality_factor = (self._get_mortality_factor(mortality_event.mortality_category))

        for product in (product_result.recommendations):

            base_premium = ((coverage_amount / 1000) * product.base_premium_rate)

            annual_premium = (base_premium * age_factor * medical_factor * mortality_factor)

            monthly_premium = (annual_premium / 12)

            rationale = [
                f"Coverage amount: ₹{coverage_amount:,.0f}",
                f"Base premium rate: {product.base_premium_rate}",
                f"Age factor: {age_factor}",
                (
                    "Medical loading factor: "
                    f"{medical_factor:.2f}"
                ),
                f"Mortality factor: {mortality_factor:.2f}"
            ]

            premium_estimates.append(

                PremiumEstimate(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    annual_premium=round(
                        annual_premium,
                        2,
                    ),
                    monthly_premium=round(
                        monthly_premium,
                        2,
                    ),
                    coverage_amount=coverage_amount,
                    rationale=rationale,
                )

            )

        premium_estimates.sort(
            key=lambda x: x.annual_premium
        )

        return premium_estimates

    # ==========================================
    # AGE FACTORS
    # ==========================================

    def _get_age_factor(
        self,
        age: int,
    ) -> float:

        if 18 <= age <= 30:
            return 0.90

        if 31 <= age <= 40:
            return 1.00

        if 41 <= age <= 50:
            return 1.20

        if 51 <= age <= 60:
            return 1.50

        return 2.00
    
    def _get_mortality_factor(self, mortality_category: str) -> float:
        factors = {
            "Low": 1.00,
            "Medium": 1.05,
            "High": 1.20,
            "Severe": 1.50
        }

        return factors.get(mortality_category, 1.00)