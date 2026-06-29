# tests/test_mortality_service.py

from domain.customer_profile import CustomerProfile

from services.customer_enrichment_service import CustomerEnrichmentService
from services.mortality_service import MortalityService


def test_mortality_service():

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

    mortality_service = (MortalityService())

    enriched = (enrichment_service.enrich(customer))

    prediction = (mortality_service.predict(enriched))

    assert (0 <= prediction.mortality_probability <= 1)

    assert (
        prediction.mortality_category
        in
        [
            "Low",
            "Medium",
            "High",
            "Severe",
        ]
    )

    assert (prediction.mortality_score >= 0)