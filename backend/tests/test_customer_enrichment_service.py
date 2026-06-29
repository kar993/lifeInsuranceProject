# tests/test_customer_enrichment_service.py

import pytest

from domain.customer_profile import CustomerProfile
from services.customer_enrichment_service import (
    CustomerEnrichmentService,
)


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def service():
    """
    Create the enrichment service once per test.
    """
    return CustomerEnrichmentService()


@pytest.fixture
def sample_customer():
    """
    Standard customer used across tests.
    """

    return CustomerProfile(
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

        diabetes=True,
        hypertension=True,
        heart_disease=False,
        respiratory_disease=False,
        cancer_history=False,

        desired_coverage=5_000_000,

        payment_mode_preference="Monthly",
        autopay_preference=True,
    )


# ==========================================================
# TESTS
# ==========================================================

def test_bmi_calculation(service):
    """
    Verify BMI calculation.
    """

    bmi = service._calculate_bmi(
        height_cm=180,
        weight_kg=81,
    )

    assert round(bmi, 2) == 25.00

def test_medical_loading(service, sample_customer):
    """
    Verify medical extra loading.
    Expected:
        Diabetes      = 0.35
        Hypertension  = 0.20
        Total         = 0.55
    """

    loading = service._calculate_medical_loading(
        sample_customer
    )

    assert loading == 0.55


def test_full_enrichment_flow(
    service,
    sample_customer,
):
    """
    Verify end-to-end enrichment.
    """

    enriched = service.enrich(
        sample_customer
    )

    # ------------------------------------------------------
    # BMI
    # ------------------------------------------------------

    assert enriched.bmi > 0
    assert enriched.bmi_category is not None
    assert enriched.bmi_multiplier > 0

    # ------------------------------------------------------
    # Occupation
    # ------------------------------------------------------

    assert enriched.occupation_risk is not None
    assert enriched.occupation_multiplier > 0

    # ------------------------------------------------------
    # Medical Loading
    # ------------------------------------------------------

    assert enriched.medical_extra_loading == 0.55

    # ------------------------------------------------------
    # Coverage Metrics
    # ------------------------------------------------------

    assert enriched.coverage_income_ratio == 5.0

    assert (
        enriched.total_requested_coverage
        == 5_500_000
    )

    assert enriched.debt_income_ratio == 0.2

    # ------------------------------------------------------
    # Age Band
    # ------------------------------------------------------

    assert enriched.age_band == "31-40"

    # ------------------------------------------------------
    # Sanity Checks
    # ------------------------------------------------------

    assert enriched.age == sample_customer.age
    assert enriched.gender == sample_customer.gender
    assert enriched.occupation == sample_customer.occupation