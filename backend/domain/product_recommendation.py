from dataclasses import dataclass


@dataclass
class ProductRecommendation:

    product_id: str

    product_name: str

    product_type: str

    risk_profile: str

    suitability_score: float

    base_premium_rate: float

    rationale: list[str]