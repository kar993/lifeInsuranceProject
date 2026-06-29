from google.adk.tools import ToolContext

from state.state_keys import REPORT_TEXT


def save_report_text_tool(
    report_text: str,
    tool_context: ToolContext,
) -> str:

    tool_context.state[REPORT_TEXT] = report_text

    return "Recommendation narrative saved."