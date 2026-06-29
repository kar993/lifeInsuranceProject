from dataclasses import dataclass

from domain.product_recommendation import (
    ProductRecommendation
)


@dataclass
class ProductRecommendationResult:

    recommendations: list[ProductRecommendation]