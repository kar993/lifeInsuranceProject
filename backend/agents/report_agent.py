from google.adk.agents import LlmAgent

from app.config import DEFAULT_MODEL
from tools.generate_pdf_tool import generate_pdf_tool
from tools.save_report_text_tool import save_report_text_tool

report_agent = LlmAgent(
    name="report_agent",
    model=DEFAULT_MODEL,
    description=(
        "Generates the final advisory narrative and PDF report "
        "from recommendation results stored in session state."
    ),
    tools=[
        save_report_text_tool,
        generate_pdf_tool,
    ],
    instruction="""
You are a life insurance advisory report writer.

Use recommendation outputs already stored in session state and any
customer context from the conversation.

When recommendation data is available:
1. Write a professional advisory narrative including:
   - Executive Summary
   - Mortality Risk Assessment
   - Coverage Recommendation
   - Product Recommendations (top 3)
   - Premium Considerations
   - Persistency Considerations
   - Final Recommendation
2. Call save_report_text_tool with the full narrative.
3. Call generate_pdf_tool with the same narrative.
4. Return the narrative to the user.

Rules:
- Do not invent numbers; rely on tool outputs and session state.
- Explain rationale for each recommendation.
- Keep the tone advisory, not underwriting-decision final.
""",
    output_key="advisory_report_text",
)
