# tests/test_premium_estimator.py

from domain.customer_profile import CustomerProfile

from services.customer_enrichment_service import CustomerEnrichmentService
from services.eligibility_engine import EligibilityEngine
from services.coverage_engine import CoverageEngine
from services.product_recommendation_engine import ProductRecommendationEngine
from services.premium_estimator_service import PremiumEstimator
from services.mortality_service import MortalityService

def test_premium_estimator():

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

    enrichment_service = CustomerEnrichmentService()
    eligibility_engine = EligibilityEngine()
    coverage_engine = CoverageEngine()
    product_engine = ProductRecommendationEngine()
    premium_estimator = PremiumEstimator()
    mortality_service = MortalityService()

    enriched = enrichment_service.enrich(customer)
    mortality_event = mortality_service.predict(enriched)
    eligibility = eligibility_engine.evaluate(enriched)
    coverage = coverage_engine.recommend(enriched)
    products = product_engine.recommend(enriched,eligibility,coverage,mortality_event)
    premiums = premium_estimator.estimate(enriched,coverage,products,mortality_event)

    # ==========================================
    # BASIC VALIDATION
    # ==========================================

    assert len(premiums) > 0

    # ==========================================
    # SORTING VALIDATION
    # ==========================================

    annual_premiums = [p.annual_premium for p in premiums]

    assert annual_premiums == sorted(annual_premiums)

    # ==========================================
    # PREMIUM CHECKS
    # ==========================================

    first = premiums[0]

    assert first.product_name is not None

    assert first.annual_premium > 0

    assert first.monthly_premium > 0

    assert (first.coverage_amount == coverage.recommended_coverage)

    # ==========================================
    # RATIONALE
    # ==========================================

    assert len(first.rationale) > 0