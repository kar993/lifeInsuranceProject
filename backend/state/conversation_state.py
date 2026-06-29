from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """
    Shared ADK conversation state.

    This state is stored inside the ADK session and acts as
    the communication layer between agents.
    """

    # Original customer input/profile
    customer_profile: dict[str, Any] | None = None

    # Full recommendation pipeline output
    recommendation_result: dict[str, Any] | None = None

    # Report builder output
    report_data: dict[str, Any] | None = None

    # LLM-generated recommendation narrative
    report_text: str | None = None

    # Generated PDF path
    pdf_path: str | None = None

    # Flags
    recommendation_generated: bool = False

    # Additional conversation metadata
    metadata: dict[str, Any] = field(default_factory=dict)