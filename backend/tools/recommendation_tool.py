from dataclasses import asdict
from datetime import datetime

from google.adk.tools import ToolContext

from domain.customer_profile import CustomerProfile

from orchestrators.recommendation_pipeline import RecommendationPipeline

from state.state_keys import (
    CUSTOMER_INPUT,
    CUSTOMER_PROFILE,
    ELIGIBILITY_RESULT,
    MORTALITY_PREDICTION,
    COVERAGE_RECOMMENDATION,
    RECOMMENDATION_RESULT,
    REPORT_DATA,
    PDF_PATH,
    RECOMMENDATION_GENERATED,
    LAST_RECOMMENDATION_TIMESTAMP,
)

pipeline = RecommendationPipeline()


def generate_recommendation_tool(
    age: int,
    gender: str,

    annual_income: float,
    marital_status: str,
    dependents: int,
    existing_life_coverage: float,
    outstanding_loans: float,

    height_cm: float,
    weight_kg: float,

    occupation: str,

    smoker_status: bool,
    alcohol_use: str,
    exercise_level: str,

    diabetes: bool,
    hypertension: bool,
    heart_disease: bool,
    respiratory_disease: bool,
    cancer_history: bool,

    desired_coverage: float,

    payment_mode_preference: str,
    autopay_preference: bool,

    tool_context: ToolContext,
) -> dict:
    """
    Run the recommendation pipeline to get the recommendations of various insurance products, coverage, and premium
    """
    raw_customer_input = {
        "age": age,
        "gender": gender,

        "annual_income": annual_income,
        "marital_status": marital_status,
        "dependents": dependents,
        "existing_life_coverage": existing_life_coverage,
        "outstanding_loans": outstanding_loans,

        "height_cm": height_cm,
        "weight_kg": weight_kg,

        "occupation": occupation,

        "smoker_status": smoker_status,
        "alcohol_use": alcohol_use,
        "exercise_level": exercise_level,

        "diabetes": diabetes,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "respiratory_disease": respiratory_disease,
        "cancer_history": cancer_history,

        "desired_coverage": desired_coverage,

        "payment_mode_preference": payment_mode_preference,
        "autopay_preference": autopay_preference,
    }

    try:

        customer_profile = CustomerProfile(
            age=age,
            gender=gender,

            annual_income=annual_income,
            marital_status=marital_status,
            dependents=dependents,
            existing_life_coverage=existing_life_coverage,
            outstanding_loans=outstanding_loans,

            height_cm=height_cm,
            weight_kg=weight_kg,

            occupation=occupation,

            smoker_status=smoker_status,
            alcohol_use=alcohol_use,
            exercise_level=exercise_level,

            diabetes=diabetes,
            hypertension=hypertension,
            heart_disease=heart_disease,
            respiratory_disease=respiratory_disease,
            cancer_history=cancer_history,

            desired_coverage=desired_coverage,

            payment_mode_preference=payment_mode_preference,
            autopay_preference=autopay_preference,
        )

        result = pipeline.run(
            customer_profile,
            generate_pdf=False,
        )
        tool_context.state[CUSTOMER_INPUT] = raw_customer_input

        tool_context.state[CUSTOMER_PROFILE] = (
            asdict(result["customer_profile"])
        )

        tool_context.state[ELIGIBILITY_RESULT] = (
            asdict(result["eligibility_result"])
        )

        tool_context.state[MORTALITY_PREDICTION] = (
            asdict(result["mortality_prediction"])
        )

        tool_context.state[COVERAGE_RECOMMENDATION] = (
            asdict(result["coverage_recommendation"])
        )

        tool_context.state[RECOMMENDATION_RESULT] = (
            asdict(result["recommendation_result"])
        )

        tool_context.state[LAST_RECOMMENDATION_TIMESTAMP] = datetime.utcnow().isoformat

        tool_context.state[REPORT_DATA] = (
            asdict(result["report_data"])
        )

        tool_context.state[PDF_PATH] = None

        tool_context.state[
            RECOMMENDATION_GENERATED
        ] = True

        mortality = result["mortality_prediction"]
        coverage = result["coverage_recommendation"]
        recommendation = result["recommendation_result"]

        return {
            "status": "success",

            "mortality_prediction":
                tool_context.state[
                    MORTALITY_PREDICTION
                ],

            "coverage_recommendation":
                tool_context.state[
                    COVERAGE_RECOMMENDATION
                ],

            "recommendation_result":
                tool_context.state[
                    RECOMMENDATION_RESULT
                ]
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }