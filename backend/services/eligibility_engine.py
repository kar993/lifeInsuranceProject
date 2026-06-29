# services/eligibility_engine.py

from pathlib import Path

import pandas as pd

from domain.enriched_customer_profile import EnrichedCustomerProfile
from domain.eligibility_result import EligibilityResult
from config.paths import DATA_DIR

class EligibilityEngine:
    """
    Evaluates underwriting eligibility rules and returns:

        ELIGIBLE
        REFER
        DECLINE

    Priority:

        DECLINE > REFER > ELIGIBLE
    """

    def __init__(
        self,
        rules_file: str = "eligibility_rules.csv"
    ):

        self.rules_df = pd.read_csv(DATA_DIR / (rules_file))

    # =====================================================
    # PUBLIC
    # =====================================================

    def evaluate(
        self,
        customer: EnrichedCustomerProfile
    ) -> EligibilityResult:

        triggered_rules = []
        explanations = []
        decisions = []

        derived_values = self._build_derived_values(
            customer
        )

        for _, rule in self.rules_df.iterrows():

            if self._rule_matches(
                customer=customer,
                derived_values=derived_values,
                field=rule["field"],
                operator=rule["operator"],
                value=rule["value"],
            ):

                triggered_rules.append(
                    rule["rule_id"]
                )

                explanations.append(
                    rule["reason"]
                )

                decisions.append(
                    rule["decision"]
                )

        final_decision = self._determine_decision(
            decisions
        )

        return EligibilityResult(
            decision=final_decision,
            triggered_rules=triggered_rules,
            explanations=explanations,
        )

    # =====================================================
    # DERIVED VALUES
    # =====================================================

    def _build_derived_values(
        self,
        customer: EnrichedCustomerProfile
    ) -> dict:

        condition_count = sum([
            customer.diabetes,
            customer.hypertension,
            customer.heart_disease,
            customer.respiratory_disease,
            customer.cancer_history,
        ])

        senior_smoker = (
            customer.smoker_status
            and customer.age > 60
        )

        return {
            "condition_count": condition_count,
            "senior_smoker": senior_smoker,
        }

    # =====================================================
    # RULE EVALUATION
    # =====================================================

    def _rule_matches(
        self,
        customer: EnrichedCustomerProfile,
        derived_values: dict,
        field: str,
        operator: str,
        value,
    ) -> bool:

        actual_value = self._get_field_value(
            customer,
            derived_values,
            field,
        )

        expected_value = self._convert_value(
            value
        )

        return self._evaluate_operator(
            actual_value,
            operator,
            expected_value,
        )

    def _get_field_value(
        self,
        customer: EnrichedCustomerProfile,
        derived_values: dict,
        field: str,
    ):

        if field in derived_values:
            return derived_values[field]

        if not hasattr(customer, field):
            raise ValueError(
                f"Unknown field in eligibility rule: {field}"
            )

        return getattr(customer, field)

    # =====================================================
    # TYPE CONVERSION
    # =====================================================

    def _convert_value(
        self,
        value,
    ):

        if pd.isna(value):
            return None

        if isinstance(value, bool):
            return value

        value = str(value).strip()

        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        try:

            if "." in value:
                return float(value)

            return int(value)

        except ValueError:
            return value

    # =====================================================
    # OPERATORS
    # =====================================================

    def _evaluate_operator(
        self,
        actual,
        operator,
        expected,
    ) -> bool:

        if operator == "<":
            return actual < expected

        if operator == "<=":
            return actual <= expected

        if operator == ">":
            return actual > expected

        if operator == ">=":
            return actual >= expected

        if operator == "==":
            return actual == expected

        if operator == "!=":
            return actual != expected

        raise ValueError(
            f"Unsupported operator: {operator}"
        )

    # =====================================================
    # DECISION PRIORITY
    # =====================================================

    def _determine_decision(
        self,
        decisions: list[str]
    ) -> str:

        if "DECLINE" in decisions:
            return "DECLINE"

        if "REFER" in decisions:
            return "REFER"

        return "ELIGIBLE"