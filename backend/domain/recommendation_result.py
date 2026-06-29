from dataclasses import dataclass

from domain.coverage_recommendation import CoverageRecommendation
from domain.product_recommendation_result import ProductRecommendationResult
from domain.premium_estimate import PremiumEstimate 
from domain.persistency_prediction import PersistencyPrediction
from domain.mortality_prediction import MortalityPrediction
from domain.recommended_product import RecommendedProduct

@dataclass
class RecommendationResult:

    mortality_prediction: MortalityPrediction
    coverage_recommendation: CoverageRecommendation
    recommended_products: list[RecommendedProduct]