# services/coverage_engine.py

from pathlib import Path

import pandas as pd

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.coverage_recommendation import CoverageRecommendation
from config.paths import DATA_DIR

class CoverageEngine:
    """
    Determines customer protection needs and
    recommends coverage ranges.

    Uses:
        coverage_recommendation_rules.csv
    """

    def __init__(
        self,
        rules_file: str = (
            "coverage_recommendation_rules.csv"
        )
    ):

        self.rules_df = pd.read_csv(DATA_DIR / (rules_file))

    # =====================================================
    # PUBLIC
    # =====================================================

    def recommend(
        self,
        customer: EnrichedCustomerProfile,
    ) -> CoverageRecommendation:

        multiplier = (
            self._get_income_multiplier(
                customer.dependents
            )
        )

        base_income_need = (
            customer.annual_income
            * multiplier
        )

        debt_protection_need = (
            customer.outstanding_loans
        )

        total_need = (
            base_income_need
            + debt_protection_need
        )

        minimum_factor = (
            self._get_factor(
                "minimum_factor"
            )
        )

        recommended_factor = (
            self._get_factor(
                "recommended_factor"
            )
        )

        stretch_factor = (
            self._get_factor(
                "stretch_factor"
            )
        )

        minimum_coverage = (
            total_need
            * minimum_factor
        )

        recommended_coverage = (
            total_need
            * recommended_factor
        )

        stretch_coverage = (
            total_need
            * stretch_factor
        )

        additional_coverage_needed = max(
            total_need
            - customer.existing_life_coverage,
            0,
        )

        coverage_gap = (
            customer.desired_coverage
            - recommended_coverage
        )

        rationale = self._build_rationale(
            customer=customer,
            multiplier=multiplier,
            total_need=total_need,
            coverage_gap=coverage_gap,
        )

        return CoverageRecommendation(

            income_multiplier=multiplier,

            base_income_need=base_income_need,

            debt_protection_need=debt_protection_need,

            total_need=total_need,

            minimum_coverage=minimum_coverage,
            recommended_coverage=recommended_coverage,
            stretch_coverage=stretch_coverage,

            existing_coverage=
                customer.existing_life_coverage,

            additional_coverage_needed=
                additional_coverage_needed,

            customer_requested_coverage=
                customer.desired_coverage,

            coverage_gap=coverage_gap,

            rationale=rationale,
        )

    # =====================================================
    # RULE LOOKUPS
    # =====================================================

    def _get_income_multiplier(
        self,
        dependents: int,
    ) -> float:

        rules = self.rules_df[
            self.rules_df["output_field"]
            == "income_multiplier"
        ]

        for _, rule in rules.iterrows():

            operator = rule["condition"]
            value = float(rule["value"])

            if self._evaluate(
                dependents,
                operator,
                value,
            ):
                return float(
                    rule["output_value"]
                )

        raise ValueError(
            "No income multiplier rule found."
        )

    def _get_factor(
        self,
        factor_name: str,
    ) -> float:

        row = self.rules_df[
            self.rules_df["output_field"]
            == factor_name
        ]

        if row.empty:
            raise ValueError(
                f"{factor_name} rule missing."
            )

        return float(
            row.iloc[0]["output_value"]
        )

    # =====================================================
    # OPERATORS
    # =====================================================

    def _evaluate(
        self,
        actual,
        operator,
        expected,
    ) -> bool:

        if operator == "==":
            return actual == expected

        if operator == ">":
            return actual > expected

        if operator == ">=":
            return actual >= expected

        if operator == "<":
            return actual < expected

        if operator == "<=":
            return actual <= expected

        raise ValueError(
            f"Unsupported operator: {operator}"
        )

    # =====================================================
    # EXPLANATIONS
    # =====================================================

    def _build_rationale(
        self,
        customer: EnrichedCustomerProfile,
        multiplier: float,
        total_need: float,
        coverage_gap: float,
    ) -> list[str]:

        rationale = []

        rationale.append(
            f"Coverage need calculated using "
            f"{multiplier:.0f}x annual income."
        )

        if customer.outstanding_loans > 0:

            rationale.append(
                "Outstanding loans were added "
                "to protection needs."
            )

        if customer.existing_life_coverage > 0:

            rationale.append(
                "Existing coverage was considered "
                "when calculating additional "
                "coverage requirements."
            )

        if coverage_gap < 0:

            rationale.append(
                "Requested coverage is below "
                "the recommended protection level."
            )

        elif coverage_gap > 0:

            rationale.append(
                "Requested coverage exceeds "
                "the recommended protection level."
            )

        else:

            rationale.append(
                "Requested coverage matches "
                "the recommended protection level."
            )

        return rationale