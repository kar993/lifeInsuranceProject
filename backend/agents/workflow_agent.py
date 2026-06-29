"""Graph-based deterministic workflow for structured profile analysis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from google.adk import Event, Workflow
from pydantic import BaseModel, Field

from domain.customer_profile import CustomerProfile
from orchestrators.recommendation_pipeline import RecommendationPipeline
from state.state_keys import (
    COVERAGE_RECOMMENDATION,
    CUSTOMER_INPUT,
    CUSTOMER_PROFILE,
    ELIGIBILITY_RESULT,
    LAST_RECOMMENDATION_TIMESTAMP,
    MORTALITY_PREDICTION,
    PDF_PATH,
    RECOMMENDATION_GENERATED,
    RECOMMENDATION_RESULT,
    REPORT_DATA,
    REPORT_TEXT,
)


class ProfilePayload(BaseModel):
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


class PipelineSummary(BaseModel):
    status: str
    eligibility_decision: str = ""
    mortality_category: str = ""
    recommended_coverage: float = 0.0
    top_product: str = ""
    top_premium: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


_pipeline = RecommendationPipeline()
_workflow_state: dict[str, Any] = {}


def validate_profile_function(raw_input: str) -> ProfilePayload:
    """Parse and validate structured profile JSON passed into the workflow."""
    import json

    payload = json.loads(raw_input)
    return ProfilePayload(**payload)


def run_recommendation_pipeline_function(
    profile: ProfilePayload,
) -> PipelineSummary:
    """Execute the full recommendation pipeline without LLM involvement."""
    from datetime import datetime

    customer = CustomerProfile(**profile.model_dump())
    result = _pipeline.run(customer, generate_pdf=False)

    _workflow_state[CUSTOMER_INPUT] = profile.model_dump()
    _workflow_state[CUSTOMER_PROFILE] = asdict(result["customer_profile"])
    _workflow_state[ELIGIBILITY_RESULT] = asdict(result["eligibility_result"])
    _workflow_state[MORTALITY_PREDICTION] = asdict(result["mortality_prediction"])
    _workflow_state[COVERAGE_RECOMMENDATION] = asdict(
        result["coverage_recommendation"]
    )
    _workflow_state[RECOMMENDATION_RESULT] = asdict(result["recommendation_result"])
    _workflow_state[REPORT_DATA] = asdict(result["report_data"])
    _workflow_state[PDF_PATH] = None
    _workflow_state[RECOMMENDATION_GENERATED] = True
    _workflow_state[LAST_RECOMMENDATION_TIMESTAMP] = datetime.utcnow().isoformat()

    recommendation = result["recommendation_result"]
    top_product = ""
    top_premium = 0.0
    if recommendation.recommended_products:
        top = recommendation.recommended_products[0]
        top_product = top.product_name
        top_premium = top.annual_premium

    return PipelineSummary(
        status="success",
        eligibility_decision=result["eligibility_result"].decision,
        mortality_category=result["mortality_prediction"].mortality_category,
        recommended_coverage=result["coverage_recommendation"].recommended_coverage,
        top_product=top_product,
        top_premium=top_premium,
        details={
            "mortality_prediction": _workflow_state[MORTALITY_PREDICTION],
            "coverage_recommendation": _workflow_state[COVERAGE_RECOMMENDATION],
            "recommendation_result": _workflow_state[RECOMMENDATION_RESULT],
            "report_data": _workflow_state[REPORT_DATA],
        },
    )


def _extract_event_message(event: Event) -> str:
    if event.message and event.message.parts:
        return "".join(part.text or "" for part in event.message.parts)
    if event.content and event.content.parts:
        return "".join(part.text or "" for part in event.content.parts)
    return ""


def format_pipeline_response_function(summary: PipelineSummary) -> Event:
    """Format deterministic pipeline output as a user-facing message."""
    if summary.status != "success":
        return Event(message="Recommendation pipeline failed.")

    message = (
        "Recommendation analysis complete.\n\n"
        f"Eligibility: {summary.eligibility_decision}\n"
        f"Mortality risk: {summary.mortality_category}\n"
        f"Recommended coverage: ₹{summary.recommended_coverage:,.0f}\n"
        f"Top product: {summary.top_product}\n"
        f"Estimated annual premium: ₹{summary.top_premium:,.0f}\n\n"
        "Use the report agent or Streamlit report tab to generate the advisory narrative."
    )
    return Event(message=message, output=summary.model_dump())


analysis_workflow = Workflow(
    name="analysis_workflow",
    edges=[
        (
            "START",
            validate_profile_function,
            run_recommendation_pipeline_function,
            format_pipeline_response_function,
        )
    ],
)


def run_structured_analysis(profile: dict[str, Any]) -> dict[str, Any]:
    """Helper used by Streamlit to run the graph workflow synchronously."""
    import json

    payload_json = json.dumps(profile)
    validated = validate_profile_function(payload_json)
    summary = run_recommendation_pipeline_function(validated)
    event = format_pipeline_response_function(summary)
    return {
        "message": _extract_event_message(event),
        "summary": summary.model_dump(),
        "session_state": dict(_workflow_state),
    }


def get_workflow_session_state() -> dict[str, Any]:
    return dict(_workflow_state)
