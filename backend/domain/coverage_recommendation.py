# domain/coverage_recommendation.py

from dataclasses import dataclass

@dataclass
class CoverageRecommendation:

    income_multiplier: float
    base_income_need: float
    debt_protection_need: float
    total_need: float

    minimum_coverage: float
    recommended_coverage: float
    stretch_coverage: float

    existing_coverage: float
    additional_coverage_needed: float
    customer_requested_coverage: float
    coverage_gap: float

    rationale: list[str]