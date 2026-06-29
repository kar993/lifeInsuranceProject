# tests/test_full_pipeline.py

from domain.customer_profile import CustomerProfile
from services.customer_enrichment_service import CustomerEnrichmentService
from services.eligibility_engine import EligibilityEngine
from services.coverage_engine import CoverageEngine
from services.product_recommendation_engine import ProductRecommendationEngine
from services.premium_estimator_service import PremiumEstimator
from services.mortality_service import MortalityService
from services.persistency_service import PersistencyService
from services.recommendation_service import RecommendationService
from services.report_builder_service import ReportBuilderService
from services.pdf_report_generator import PDFReportGenerator

def test_full_pipeline():

    # =====================================================
    # INPUT CUSTOMER
    # =====================================================

    customer = CustomerProfile(
        age=68,
        gender="Male",

        annual_income=1_500_000,
        marital_status="Married",
        dependents=2,

        existing_life_coverage=1_000_000,
        outstanding_loans=2_000_000,

        height_cm=175,
        weight_kg=80,

        occupation="Software Engineer",

        smoker_status=True,
        alcohol_use="Occasional",
        exercise_level="Moderate",

        diabetes=False,
        hypertension=False,
        heart_disease=True,
        respiratory_disease=False,
        cancer_history=False,

        desired_coverage=60_000_000,

        payment_mode_preference="Monthly",
        autopay_preference=True,
    )

    # =====================================================
    # SERVICES
    # =====================================================

    enrichment_service = CustomerEnrichmentService()
    eligibility_engine = EligibilityEngine()
    coverage_engine = CoverageEngine()
    product_engine = ProductRecommendationEngine()
    premium_estimator = PremiumEstimator()
    mortality_service = MortalityService()
    persistency_service = PersistencyService()
    recommendation_service = RecommendationService()

    # =====================================================
    # ENRICHMENT
    # =====================================================

    enriched_customer = (enrichment_service.enrich(customer))

    assert enriched_customer.bmi > 0

    assert (enriched_customer.occupation_risk is not None)

    # =====================================================
    # ELIGIBILITY
    # =====================================================

    eligibility_result = (eligibility_engine.evaluate(enriched_customer))

    assert (
        eligibility_result.decision
        in [
            "ELIGIBLE",
            "REFER",
            "DECLINE",
        ]
    )

    # This profile should normally pass

    assert (eligibility_result.decision == "REFER")

    # =====================================================
    # MORTALITY PREDICTION
    # =====================================================

    mortality_event = (mortality_service.predict(enriched_customer))

    assert (0 <= mortality_event.mortality_probability <= 1)

    assert (
        mortality_event.mortality_category
        in
        [
            "Low",
            "Medium",
            "High",
            "Severe",
        ]
    )

    assert (mortality_event.mortality_score >= 0)
    
    # =====================================================
    # COVERAGE
    # =====================================================

    coverage_result = (coverage_engine.recommend(enriched_customer))

    assert (coverage_result.recommended_coverage > 0)

    assert (coverage_result.minimum_coverage < coverage_result.recommended_coverage)

    assert (coverage_result.stretch_coverage > coverage_result.recommended_coverage)

    assert (coverage_result.additional_coverage_needed >= 0)

    # =====================================================
    # COVERAGE CONSISTENCY
    # =====================================================

    assert (coverage_result.total_need == coverage_result.base_income_need + coverage_result.debt_protection_need)

    # =====================================================
    # CUSTOMER REQUEST
    # =====================================================

    assert (coverage_result.customer_requested_coverage == customer.desired_coverage)

    # =====================================================
    # RATIONALE
    # =====================================================

    assert (len(coverage_result.rationale) > 0)

    # =====================================================
    # PRODUCT RECOMMENDATION
    # =====================================================

    product_result = (
        product_engine.recommend(
            enriched_customer,
            eligibility_result,
            coverage_result,
            mortality_event,
        )
    )

    assert (len(product_result.recommendations) > 0)

    scores = [p.suitability_score for p in product_result.recommendations]

    assert scores == sorted(scores, reverse=True,)

    top_product = (product_result.recommendations[0])

    assert (top_product.product_name is not None)

    assert (top_product.suitability_score > 0)

    assert (len(top_product.rationale) > 0)

    # =====================================================
    # PREMIUM ESTIMATION
    # =====================================================

    premium_result = (
        premium_estimator.estimate(
            enriched_customer,
            coverage_result,
            product_result,
            mortality_event,
        )
    )

    persistency_event = (
        persistency_service.predict(
            enriched_customer,
            coverage_result,
            product_result,
            premium_result,
        )
    )

    print(type(mortality_event))
    print(type(premium_result))
    print(type(persistency_event))

    recommendation_result = (
        recommendation_service.assemble(
            coverage_result,
            product_result,
            premium_result,
            persistency_event,
            mortality_event,
        )
    )

    report_builder = (ReportBuilderService())
    report_data = (report_builder.build(customer,eligibility_result,recommendation_result,))
    pdf_generator = (PDFReportGenerator())
    pdf_generator.generate(report_data,"sample_advisory_report.pdf",)


    assert len(premium_result) > 0

    annual_premiums = [p.annual_premium for p in premium_result]

    assert annual_premiums == sorted(annual_premiums)

    first_premium = premium_result[0]

    assert (first_premium.annual_premium > 0)

    assert (first_premium.monthly_premium > 0)

    assert (first_premium.coverage_amount == coverage_result.recommended_coverage)

    assert (len(first_premium.rationale) > 0)

    assert len(persistency_event) > 0

    for event in persistency_event:
        assert (0 <= event.lapse_probability <= 1)

    assert (recommendation_result is not None)

    assert (recommendation_result.mortality_prediction is not None)

    assert (recommendation_result.coverage_recommendation is not None)

    assert (len(recommendation_result.recommended_products) > 0)