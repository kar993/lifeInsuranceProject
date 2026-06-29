from dataclasses import dataclass


@dataclass
class RecommendedProduct:

    product_id: str
    product_name: str
    product_type: str
    suitability_score: float
    annual_premium: float
    monthly_premium: float
    lapse_probability: float
    lapse_category: str
    rationale: list[str]