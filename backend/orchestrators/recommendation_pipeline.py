from domain.customer_profile import CustomerProfile

from services.customer_enrichment_service import CustomerEnrichmentService
from services.eligibility_engine import EligibilityEngine
from services.mortality_service import MortalityService
from services.coverage_engine import CoverageEngine
from services.product_recommendation_engine import ProductRecommendationEngine
from services.premium_estimator_service import PremiumEstimator
from services.persistency_service import PersistencyService
from services.recommendation_service import RecommendationService
from services.report_builder_service import ReportBuilderService
from services.pdf_report_generator import PDFReportGenerator


class RecommendationPipeline:

    def __init__(self):
        self.enrichment_service = CustomerEnrichmentService()
        self.eligibility_engine = EligibilityEngine()
        self.mortality_service = MortalityService()
        self.coverage_engine = CoverageEngine()
        self.product_engine = ProductRecommendationEngine()
        self.premium_estimator = PremiumEstimator()
        self.persistency_service = PersistencyService()
        self.recommendation_service = RecommendationService()
        self.report_builder = ReportBuilderService()
        self.pdf_generator = PDFReportGenerator()

    def run(self,customer_profile: CustomerProfile, generate_pdf: bool = False,):

        enriched_profile = (self.enrichment_service.enrich(customer_profile))

        eligibility_result = (self.eligibility_engine.evaluate(enriched_profile))

        mortality_prediction = (self.mortality_service.predict(enriched_profile))

        coverage_recommendation = (self.coverage_engine.recommend(enriched_profile))

        product_result = self.product_engine.recommend(
            enriched_profile,
            eligibility_result,
            coverage_recommendation,
            mortality_prediction,
        )

        premium_estimates = self.premium_estimator.estimate(
            enriched_profile,
            coverage_recommendation,
            product_result,
            mortality_prediction,
        )

        persistency_predictions = self.persistency_service.predict(
            enriched_profile,
            coverage_recommendation,
            product_result,
            premium_estimates,
        )

        recommendation_result = self.recommendation_service.assemble(
            coverage_recommendation,
            product_result,
            premium_estimates,
            persistency_predictions,
            mortality_prediction,
        )

        report_data = self.report_builder.build(
            enriched_profile,
            eligibility_result,
            recommendation_result,
        )

        pdf_path = None

        if generate_pdf:
            pdf_path = self.pdf_generator.generate(report_data)

        return {
            "customer_profile": enriched_profile,
            "eligibility_result": eligibility_result,
            "mortality_prediction": mortality_prediction,
            "coverage_recommendation": coverage_recommendation,
            "recommendation_result": recommendation_result,
            "report_data": report_data,
            "pdf_path": pdf_path,
        }