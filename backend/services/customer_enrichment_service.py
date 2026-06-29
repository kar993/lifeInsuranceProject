# services/customer_enrichment_service.py

from pathlib import Path

import pandas as pd

from domain.customer_profile import CustomerProfile
from domain.enriched_customer_profile import EnrichedCustomerProfile
from config.paths import DATA_DIR

class CustomerEnrichmentService:
    """
    Responsible for converting a CustomerProfile into an
    EnrichedCustomerProfile by deriving underwriting,
    mortality-model, and recommendation features.
    """

    def __init__(
        self,
        data_dir: str = "data"
    ):

        data_path = Path(data_dir)

        self.occupation_df = pd.read_csv(
            DATA_DIR / "01_occupation_risk_table.csv"
        )

        self.bmi_df = pd.read_csv(
            DATA_DIR / "02_bmi_risk_table.csv"
        )

        self.medical_df = pd.read_csv(
            DATA_DIR / "03b_medical_condition_loading.csv"
        )

        self.medical_loading_map = self._build_medical_loading_map()

    # =====================================================
    # PUBLIC
    # =====================================================

    def enrich(
        self,
        customer: CustomerProfile
    ) -> EnrichedCustomerProfile:

        bmi = self._calculate_bmi(
            customer.height_cm,
            customer.weight_kg
        )

        bmi_category, bmi_multiplier = (
            self._get_bmi_info(bmi)
        )

        occupation_risk, occupation_multiplier = (
            self._get_occupation_info(
                customer.occupation
            )
        )

        medical_extra_loading = (
            self._calculate_medical_loading(
                customer
            )
        )

        age_band = self._get_age_band(
            customer.age
        )

        if customer.annual_income > 0:

            coverage_income_ratio = (
                customer.desired_coverage
                / customer.annual_income
            )

            debt_income_ratio = (
                customer.outstanding_loans
                / customer.annual_income
            )

        else:

            coverage_income_ratio = 0.0
            debt_income_ratio = 0.0

        total_requested_coverage = (
            customer.existing_life_coverage
            + customer.desired_coverage
        )

        return EnrichedCustomerProfile(

            # =================================================
            # ORIGINAL CUSTOMER FIELDS
            # =================================================

            age=customer.age,
            gender=customer.gender,

            annual_income=customer.annual_income,
            marital_status=customer.marital_status,
            dependents=customer.dependents,

            existing_life_coverage=customer.existing_life_coverage,
            outstanding_loans=customer.outstanding_loans,

            height_cm=customer.height_cm,
            weight_kg=customer.weight_kg,

            occupation=customer.occupation,

            smoker_status=customer.smoker_status,
            alcohol_use=customer.alcohol_use,
            exercise_level=customer.exercise_level,

            diabetes=customer.diabetes,
            hypertension=customer.hypertension,
            heart_disease=customer.heart_disease,
            respiratory_disease=customer.respiratory_disease,
            cancer_history=customer.cancer_history,

            desired_coverage=customer.desired_coverage,

            payment_mode_preference=customer.payment_mode_preference,
            autopay_preference=customer.autopay_preference,

            # =================================================
            # ENRICHED FIELDS
            # =================================================

            age_band=age_band,

            bmi=bmi,
            bmi_category=bmi_category,
            bmi_multiplier=bmi_multiplier,

            occupation_risk=occupation_risk,
            occupation_multiplier=occupation_multiplier,

            medical_extra_loading=medical_extra_loading,

            coverage_income_ratio=coverage_income_ratio,
            total_requested_coverage=total_requested_coverage,
            debt_income_ratio=debt_income_ratio,
        )

    # =====================================================
    # BMI
    # =====================================================

    @staticmethod
    def _calculate_bmi(
        height_cm: float,
        weight_kg: float
    ) -> float:

        if height_cm <= 0:
            raise ValueError(
                "Height must be greater than zero."
            )

        height_m = height_cm / 100

        bmi = weight_kg / (height_m ** 2)

        return round(bmi, 2)

    def _get_bmi_info(
        self,
        bmi: float
    ) -> tuple[str, float]:

        row = self.bmi_df[
            (self.bmi_df["bmi_min"] <= bmi)
            &
            (self.bmi_df["bmi_max"] > bmi)
        ]

        if row.empty:
            raise ValueError(
                f"No BMI category found for BMI={bmi}"
            )

        row = row.iloc[0]

        return (
            str(row["category"]),
            float(row["mortality_multiplier"])
        )

    # =====================================================
    # OCCUPATION
    # =====================================================

    def _get_occupation_info(
        self,
        occupation: str
    ) -> tuple[str, float]:

        row = self.occupation_df[
            self.occupation_df["occupation"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            occupation.strip().lower()
        ]

        if row.empty:
            raise ValueError(
                f"Unknown occupation: {occupation}"
            )

        row = row.iloc[0]

        return (
            str(row["risk_level"]),
            float(row["risk_multiplier"])
        )

    # =====================================================
    # MEDICAL LOADING
    # =====================================================

    def _build_medical_loading_map(self) -> dict:

        return {
            str(condition).strip():
            float(loading)
            for condition, loading in zip(
                self.medical_df["medical_condition"],
                self.medical_df["extra_loading"]
            )
        }

    def _calculate_medical_loading(
        self,
        customer: CustomerProfile
    ) -> float:

        total = 0.0

        if customer.diabetes:
            total += self.medical_loading_map.get(
                "Diabetes",
                0.0
            )

        if customer.hypertension:
            total += self.medical_loading_map.get(
                "Hypertension",
                0.0
            )

        if customer.heart_disease:
            total += self.medical_loading_map.get(
                "Heart Disease",
                0.0
            )

        if customer.respiratory_disease:
            total += self.medical_loading_map.get(
                "Asthma",
                0.0
            )

        if customer.cancer_history:
            total += self.medical_loading_map.get(
                "Cancer History",
                0.0
            )

        return round(total, 2)

    # =====================================================
    # AGE BAND
    # =====================================================

    @staticmethod
    def _get_age_band(
        age: int
    ) -> str:

        if 18 <= age <= 30:
            return "18-30"

        if 31 <= age <= 40:
            return "31-40"

        if 41 <= age <= 50:
            return "41-50"

        if 51 <= age <= 60:
            return "51-60"

        return "61+"