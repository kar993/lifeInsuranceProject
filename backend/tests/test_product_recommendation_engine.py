# tests/test_product_recommendation_engine.py

from domain.customer_profile import CustomerProfile

from services.customer_enrichment_service import CustomerEnrichmentService
from services.eligibility_engine import EligibilityEngine
from services.coverage_engine import CoverageEngine
from services.product_recommendation_engine import ProductRecommendationEngine
from services.mortality_service import MortalityService


def test_product_recommendation_engine():

    customer = CustomerProfile(
        age=35,
        gender="Male",

        annual_income=1_000_000,
        marital_status="Married",
        dependents=2,

        existing_life_coverage=500_000,
        outstanding_loans=200_000,

        height_cm=180,
        weight_kg=81,

        occupation="Software Engineer",

        smoker_status=False,
        alcohol_use="Occasional",
        exercise_level="Moderate",

        diabetes=False,
        hypertension=False,
        heart_disease=False,
        respiratory_disease=False,
        cancer_history=False,

        desired_coverage=5_000_000,

        payment_mode_preference="Monthly",
        autopay_preference=True,
    )

    enrichment_service = (CustomerEnrichmentService())
    eligibility_engine = (EligibilityEngine())
    coverage_engine = (CoverageEngine())
    product_engine = (ProductRecommendationEngine())
    mortality_service = (MortalityService())

    enriched = (enrichment_service.enrich(customer))
    eligibility = (eligibility_engine.evaluate(enriched))
    coverage = (coverage_engine.recommend(enriched))
    mortality_event = mortality_service.predict(enriched)

    result = (
        product_engine.recommend(
            enriched,
            eligibility,
            coverage,
            mortality_event
        )
    )

    # ==================================================
    # BASIC VALIDATION
    # ==================================================

    assert len(result.recommendations) > 0

    # ==================================================
    # SORTING VALIDATION
    # ==================================================

    scores = [
        p.suitability_score
        for p in result.recommendations
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ==================================================
    # TOP RECOMMENDATION
    # ==================================================

    top_product = result.recommendations[0]

    assert top_product.product_name is not None
    assert top_product.product_type is not None
    assert top_product.suitability_score > 0

    # ==================================================
    # RATIONALE
    # ==================================================

    assert len(top_product.rationale) > 0