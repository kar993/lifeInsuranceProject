from dataclasses import dataclass

@dataclass
class EnrichedCustomerProfile:

    # ==================================
    # Original Customer Data
    # ==================================

    age: int
    gender: str

    annual_income: float
    marital_status: str
    dependents: int

    existing_life_coverage: float
    outstanding_loans: float

    height_cm: float
    weight_kg: float

    occupation: str

    smoker_status: bool
    alcohol_use: str
    exercise_level: str

    diabetes: bool
    hypertension: bool
    heart_disease: bool
    respiratory_disease: bool
    cancer_history: bool

    desired_coverage: float

    payment_mode_preference: str
    autopay_preference: bool

    # ==================================
    # Derived Age Information
    # ==================================
    age_band: str

    # ==================================
    # Derived BMI Information
    # ==================================

    bmi: float
    bmi_category: str
    bmi_multiplier: float

    # ==================================
    # Occupation Enrichment
    # ==================================

    occupation_risk: str
    occupation_multiplier: float

    # ==================================
    # Medical Enrichment
    # ==================================

    medical_extra_loading: float

    # ==================================
    # Financial Ratios
    # ==================================

    coverage_income_ratio: float

    total_requested_coverage: float

    debt_income_ratio: float