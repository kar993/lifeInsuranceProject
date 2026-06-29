from google.adk.agents import LlmAgent

from app.config import DEFAULT_MODEL
from agents.intake_agent import intake_agent
from agents.recommendation_agent import recommendation_agent
from agents.report_agent import report_agent

coordinator_agent = LlmAgent(
    name="life_insurance_coordinator",
    model=DEFAULT_MODEL,
    description=(
        "Coordinates life insurance advisory workflow across intake, "
        "recommendation analysis, and report generation."
    ),
    instruction="""
You are the coordinator for a life insurance advisory system.

Route the conversation to the right specialist:

1. Use intake_agent when the customer profile is incomplete or the user
   is still providing personal, financial, lifestyle, or medical details.

2. Use recommendation_agent when the customer profile is complete and
   the user wants product, coverage, premium, or risk analysis.
   The recommendation agent will call backend tools for calculations.

3. Use report_agent when recommendations already exist and the user wants
   a written advisory report or PDF.

Always maintain continuity across the conversation.
Never invent insurance calculations; rely on specialist agents and tools.
""",
    sub_agents=[
        intake_agent,
        recommendation_agent,
        report_agent,
    ],
)
