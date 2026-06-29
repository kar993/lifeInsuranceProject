from dataclasses import dataclass

@dataclass
class CustomerProfile:

    # Demographics
    age: int
    gender: str

    # Financial
    annual_income: float
    marital_status: str
    dependents: int
    existing_life_coverage: float
    outstanding_loans: float

    # Physical
    height_cm: float
    weight_kg: float

    # Occupation
    occupation: str

    # Lifestyle
    smoker_status: bool
    alcohol_use: str
    exercise_level: str

    # Medical
    diabetes: bool
    hypertension: bool
    heart_disease: bool
    respiratory_disease: bool
    cancer_history: bool

    # Customer Intent
    desired_coverage: float

    # Payment
    payment_mode_preference: str
    autopay_preference: bool