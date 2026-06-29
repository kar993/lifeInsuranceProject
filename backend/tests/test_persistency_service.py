import pytest

from services.customer_enrichment_service import CustomerEnrichmentService
from services.coverage_engine import CoverageEngine
from services.eligibility_engine import EligibilityEngine
from services.mortality_service import MortalityService
from services.product_recommendation_engine import ProductRecommendationEngine
from services.premium_estimator_service import PremiumEstimator
from services.persistency_service import PersistencyService
from domain.customer_profile import CustomerProfile


def test_persistency_service():

    customer = CustomerProfile(
        age=35,
        gender="Male",

        annual_income=1200000,
        marital_status="Married",
        dependents=2,

        existing_life_coverage=1000000,
        outstanding_loans=2000000,

        height_cm=175,
        weight_kg=78,

        occupation="Software Engineer",

        smoker_status=False,
        alcohol_use="Moderate",
        exercise_level="Medium",

        diabetes=False,
        hypertension=False,
        heart_disease=False,
        respiratory_disease=False,
        cancer_history=False,

        desired_coverage=10000000,

        payment_mode_preference="Annual",
        autopay_preference=True,
    )

    enrichment_service = (CustomerEnrichmentService())
    eligibility_engine = (EligibilityEngine())
    mortality_service = (MortalityService())
    coverage_engine = (CoverageEngine())
    product_engine = (ProductRecommendationEngine())
    premium_estimator = (PremiumEstimator())
    persistency_service = (PersistencyService())

    enriched = (enrichment_service.enrich(customer))
    eligibility = (eligibility_engine.evaluate(enriched))
    mortality_event = (mortality_service.predict(enriched))
    coverage = (coverage_engine.recommend(enriched))
    products = (
        product_engine.recommend(
            enriched,
            eligibility,
            coverage,
            mortality_event,
        )
    )
    premiums = (
        premium_estimator.estimate(
            enriched,
            coverage,
            products,
            mortality_event,
        )
    )
    persistency_predictions = (
        persistency_service.predict(
            enriched,
            coverage,
            products,
            premiums,
        )
    )

    assert len(persistency_predictions) > 0

    for prediction in (persistency_predictions):

        assert (0 <= prediction.lapse_probability <= 1)

        assert (
            prediction.lapse_category
            in [
                "Low",
                "Medium",
                "High",
                "Severe",
            ]
        )