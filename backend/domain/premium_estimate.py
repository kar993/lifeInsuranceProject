# domain/premium_estimate.py

from dataclasses import dataclass


@dataclass
class PremiumEstimate:

    product_id: str

    product_name: str

    annual_premium: float

    monthly_premium: float

    coverage_amount: float

    rationale: list[str]