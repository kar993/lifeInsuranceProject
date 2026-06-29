"""ADK Runner integration with session, state, memory, and events."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk import Runner
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

from agents.coordinator_agent import coordinator_agent
from app.config import APP_NAME, DEFAULT_MODEL, DEFAULT_USER_ID, FALLBACK_MODELS
from state.state_manager import StateManager

MAX_TRANSIENT_RETRIES = 4
RETRY_BASE_DELAY_SEC = 2
_TRANSIENT_ERROR_MARKERS = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "HIGH DEMAND",
    "OVERLOADED",
)


def _is_transient_model_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _models_to_try() -> list[str]:
    models: list[str] = []
    for model in [DEFAULT_MODEL, *FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)
    return models


def _set_all_agent_models(model: str) -> None:
    coordinator_agent.model = model
    for agent in coordinator_agent.sub_agents:
        agent.model = model


@dataclass
class AgentResponse:
    text: str
    events: list[dict[str, Any]] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)


class AdvisoryAgentRunner:
    """Wraps ADK Runner with in-memory session and memory services."""

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()
        self.memory_service = InMemoryMemoryService()
        self.runner = Runner(
            agent=coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service,
            memory_service=self.memory_service,
        )

    async def create_session(
        self,
        user_id: str = DEFAULT_USER_ID,
        initial_state: dict[str, Any] | None = None,
    ) -> Session:
        return await self.session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            state=initial_state or {},
        )

    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Session | None:
        return await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

    async def list_sessions(self, user_id: str = DEFAULT_USER_ID) -> list[Session]:
        response = await self.session_service.list_sessions(
            app_name=APP_NAME,
            user_id=user_id,
        )
        return response.sessions

    async def delete_session(self, user_id: str, session_id: str) -> None:
        await self.session_service.delete_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

    async def send_message(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str | None = None,
    ) -> AgentResponse:
        if not session_id:
            session = await self.create_session(user_id=user_id)
            session_id = session.id
        else:
            session = await self.get_session(user_id=user_id, session_id=session_id)
            if not session:
                session = await self.session_service.create_session(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    state={},
                )

        last_error: Exception | None = None
        for model in _models_to_try():
            _set_all_agent_models(model)
            for attempt in range(MAX_TRANSIENT_RETRIES):
                try:
                    return await self._run_message_once(
                        message=message,
                        user_id=user_id,
                        session_id=session_id,
                    )
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_model_error(exc):
                        raise
                    if attempt < MAX_TRANSIENT_RETRIES - 1:
                        await asyncio.sleep(RETRY_BASE_DELAY_SEC * (2**attempt))
                        continue
                    break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Agent request failed without a response.")

    async def _run_message_once(
        self,
        message: str,
        user_id: str,
        session_id: str,
    ) -> AgentResponse:
        content = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )

        response_text = ""
        captured_events: list[dict[str, Any]] = []

        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            event_record = {
                "author": getattr(event, "author", None),
                "partial": getattr(event, "partial", False),
            }
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        if not getattr(event, "partial", False):
                            response_text = part.text
                        event_record["text"] = part.text
            captured_events.append(event_record)

        session = await self.get_session(user_id, session_id)
        session_state = dict(session.state) if session else {}

        if session and StateManager.recommendation_exists(session):
            await self.memory_service.add_session_to_memory(session)

        return AgentResponse(
            text=response_text,
            events=captured_events,
            session_state=session_state,
        )

    def send_message_sync(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str | None = None,
    ) -> AgentResponse:
        return asyncio.run(
            self.send_message(
                message=message,
                user_id=user_id,
                session_id=session_id,
            )
        )

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())
