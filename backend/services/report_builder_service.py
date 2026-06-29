from domain.report_data import ReportData


class ReportBuilderService:

    def build(self,customer,eligibility,recommendation_result,) -> ReportData:

        return ReportData(
            customer_name="Customer",
            eligibility_status=eligibility.decision,
            mortality_probability=recommendation_result.mortality_prediction.mortality_probability,
            mortality_category=recommendation_result.mortality_prediction.mortality_category,
            total_protection_need=recommendation_result.coverage_recommendation.total_need,
            recommended_coverage=recommendation_result.coverage_recommendation.recommended_coverage,
            stretch_coverage=recommendation_result.coverage_recommendation.stretch_coverage,
            recommended_products=recommendation_result.recommended_products,
            advisory_summary="",
        )