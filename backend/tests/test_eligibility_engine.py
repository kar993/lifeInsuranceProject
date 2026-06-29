# tests/test_eligibility_engine.py

from domain.customer_profile import CustomerProfile
from services.customer_enrichment_service import (
    CustomerEnrichmentService,
)
from services.eligibility_engine import (
    EligibilityEngine,
)


def test_eligibility_engine():

    customer = CustomerProfile(
        age=65,
        gender="Male",

        annual_income=1_000_000,
        marital_status="Married",
        dependents=2,

        existing_life_coverage=500_000,
        outstanding_loans=200_000,

        height_cm=180,
        weight_kg=81,

        occupation="Software Engineer",

        smoker_status=True,
        alcohol_use="Occasional",
        exercise_level="Moderate",

        diabetes=True,
        hypertension=True,
        heart_disease=True,
        respiratory_disease=False,
        cancer_history=False,

        desired_coverage=60_000_000,

        payment_mode_preference="Monthly",
        autopay_preference=True,
    )

    enrichment_service = CustomerEnrichmentService()
    eligibility_engine = EligibilityEngine()

    enriched = enrichment_service.enrich(customer)

    result = eligibility_engine.evaluate(enriched)

    assert result.decision == "REFER"
    assert len(result.triggered_rules) > 0