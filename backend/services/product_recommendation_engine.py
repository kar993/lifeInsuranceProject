# services/product_recommendation_engine.py

from pathlib import Path
import pandas as pd

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.eligibility_result import EligibilityResult
from domain.coverage_recommendation import CoverageRecommendation
from domain.product_recommendation import ProductRecommendation
from domain.product_recommendation_result import ProductRecommendationResult
from domain.mortality_prediction import MortalityPrediction
from config.paths import DATA_DIR

class ProductRecommendationEngine:
    """
    Recommends suitable insurance products based on:

        - Eligibility
        - Coverage needs
        - Customer profile

    Returns all valid products ranked by
    suitability score.
    """

    def __init__(
        self,
        catalog_file: str = "product_catalog_sc_new.csv",
    ):
        self.catalog_df = pd.read_csv(DATA_DIR / catalog_file)

    # =====================================================
    # PUBLIC
    # =====================================================

    def recommend(
        self,
        customer: EnrichedCustomerProfile,
        eligibility_result: EligibilityResult,
        coverage_result: CoverageRecommendation,
        mortality_event: MortalityPrediction,
    ) -> ProductRecommendationResult:

        if eligibility_result.decision == "DECLINE":

            return ProductRecommendationResult(recommendations=[])

        recommendations = []

        filtered_products = (self._filter_products(customer,coverage_result,))

        for _, product in filtered_products.iterrows():

            score, rationale = self._score_product(customer,coverage_result,mortality_event,product,)

            recommendations.append(

                ProductRecommendation(
                    product_id=product["product_id"],
                    product_name=product["product_name"],
                    product_type=product["product_type"],
                    risk_profile=product["risk_profile"],
                    suitability_score=score,
                    base_premium_rate=float(
                        product[
                            "base_premium_rate"
                        ]
                    ),
                    rationale=rationale,
                )

            )

        recommendations.sort(key=lambda x: x.suitability_score,reverse=True,)

        return ProductRecommendationResult(recommendations=recommendations)

    # =====================================================
    # FILTERING
    # =====================================================

    def _filter_products(
        self,
        customer: EnrichedCustomerProfile,
        coverage_result: CoverageRecommendation,
    ) -> pd.DataFrame:

        recommended_coverage = coverage_result.recommended_coverage

        df = self.catalog_df.copy()

        df = df[(df["min_age"] <= customer.age) & (df["max_age"] >= customer.age)]

        df = df[(df["min_coverage"] <= recommended_coverage) & (df["max_coverage"] >= recommended_coverage)]

        return df

    # =====================================================
    # SCORING
    # =====================================================

    def _score_product(
        self,
        customer: EnrichedCustomerProfile,
        coverage_result: CoverageRecommendation,
        mortality_event: MortalityPrediction,
        product,
    ):

        score = 50

        rationale = []

        product_type = (str(product["product_type"]).strip())

        # ==========================================
        # DEPENDENTS
        # ==========================================

        if customer.dependents >= 2:

            if product_type == "Term Life":
                score += 20
                rationale.append("Suitable for protecting multiple dependents.")

            elif product_type == "Whole Life":
                score += 15
                rationale.append("Provides long-term family protection.")

        # ==========================================
        # OUTSTANDING LOANS
        # ==========================================

        if customer.outstanding_loans > 0:

            if product_type == "Term Life":
                score += 15
                rationale.append("Effective for protecting outstanding loans.")

        # ==========================================
        # LARGE COVERAGE NEED
        # ==========================================

        if (
            coverage_result.recommended_coverage
            >= 10_000_000
        ):

            if product_type == "Term Life":
                score += 20
                rationale.append("Efficient solution for large coverage needs.")

        # ==========================================
        # YOUNG CUSTOMER
        # ==========================================

        if customer.age <= 35:

            if product_type == "ULIP":
                score += 15
                rationale.append("Long investment horizon supports market-linked products.")

        # ==========================================
        # MID CAREER
        # ==========================================

        elif customer.age <= 50:

            if product_type == "Endowment":
                score += 10
                rationale.append("Balanced protection and savings.")

            elif product_type == "Money Back":
                score += 10
                rationale.append("Periodic payouts may suit mid-career financial planning.")

        # ==========================================
        # OLDER CUSTOMER
        # ==========================================

        else:

            if product_type == "Whole Life":
                score += 15
                rationale.append("Lifetime protection may be valuable at this stage.")

        # ==========================================
        # EXISTING COVERAGE
        # ==========================================

        if (
            customer.existing_life_coverage
            < coverage_result.total_need
        ):

            if product_type == "Term Life":
                score += 10
                rationale.append("Helps close the protection gap cost-effectively.")

        # ==========================================
        # MORTALITY RISK
        # ==========================================

        if mortality_event.mortality_category == "High":

            if product_type == "Term Life":
                score += 10

                rationale.append("Higher mortality risk increases the importance of pure protection.")

            elif product_type == "Whole Life":
                score += 5

                rationale.append("Provides long-term protection under elevated mortality risk.")

            elif product_type == "ULIP":
                score -= 5

                rationale.append("Investment-focused products are less suitable under elevated mortality risk.")


        elif mortality_event.mortality_category == "Severe":

            if product_type == "Term Life":
                score += 20

                rationale.append("Severe mortality risk strongly favors protection-focused coverage.")

            elif product_type == "Whole Life":
                score += 10

                rationale.append("Lifetime protection remains valuable under severe mortality risk.")

            elif product_type == "ULIP":
                score -= 15

                rationale.append("Investment-linked products become less suitable under severe mortality risk.")

            elif product_type == "Money Back":
                score -= 10

                rationale.append("Protection needs take priority over periodic payout features.")

            elif product_type == "Endowment":
                score -= 10

                rationale.append("Protection needs take priority over savings-oriented products.")

        # ==========================================
        # DEFAULT EXPLANATION
        # ==========================================

        if not rationale:

            rationale.append("Product satisfies eligibility and coverage requirements.")

        return score, rationale