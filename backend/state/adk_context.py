from dataclasses import dataclass

from google.adk.sessions import Session


@dataclass
class AgentContext:
    """
    Convenience wrapper around
    ADK Session.
    """

    session: Session