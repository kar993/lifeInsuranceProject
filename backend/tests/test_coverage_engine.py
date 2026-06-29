# tests/test_coverage_engine.py

from domain.customer_profile import CustomerProfile

from services.customer_enrichment_service import (
    CustomerEnrichmentService,
)

from services.coverage_engine import (
    CoverageEngine,
)


def test_coverage_engine():

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
    coverage_engine = CoverageEngine()

    enriched_customer = (
        enrichment_service.enrich(customer)
    )

    result = coverage_engine.recommend(
        enriched_customer
    )

    # =====================================================
    # Income Multiplier
    # =====================================================

    assert result.income_multiplier == 12

    # =====================================================
    # Income Need
    # =====================================================

    assert result.base_income_need == 12_000_000

    # =====================================================
    # Debt Need
    # =====================================================

    assert result.debt_protection_need == 200_000

    # =====================================================
    # Total Need
    # =====================================================

    assert result.total_need == 12_200_000

    # =====================================================
    # Coverage Ranges
    # =====================================================

    assert result.minimum_coverage == 9_760_000
    assert result.recommended_coverage == 12_200_000
    assert result.stretch_coverage == 14_640_000

    # =====================================================
    # Existing Coverage
    # =====================================================

    assert result.existing_coverage == 500_000

    # =====================================================
    # Additional Coverage Needed
    # =====================================================

    assert result.additional_coverage_needed == 11_700_000

    # =====================================================
    # Requested Coverage
    # =====================================================

    assert result.customer_requested_coverage == 5_000_000

    # =====================================================
    # Coverage Gap
    # =====================================================

    assert result.coverage_gap == -7_200_000

    # =====================================================
    # Rationale
    # =====================================================

    assert len(result.rationale) > 0