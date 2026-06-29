from dataclasses import dataclass


@dataclass
class ReportData:

    customer_name: str

    eligibility_status: str

    mortality_probability: float
    mortality_category: str

    total_protection_need: float
    recommended_coverage: float
    stretch_coverage: float

    recommended_products: list

    advisory_summary: str