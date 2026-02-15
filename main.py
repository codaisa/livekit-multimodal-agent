"""
---
title: Mike Voice Agent
category: education
tags: [education, multilingual, livekit, voice-agent]
---
"""

import logging
import os
import json
import base64
import asyncio
import requests
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, RoomInputOptions, metrics
from livekit.agents.voice import Agent, AgentSession, MetricsCollectedEvent
from livekit.plugins import openai, silero, deepgram, elevenlabs, google, noise_cancellation
from livekit.agents.telemetry import set_tracer_provider
from livekit import rtc

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue
import jwt

load_dotenv('.env')

logger = logging.getLogger("mike-voice-agent")
logger.setLevel(logging.INFO)

MIKE_B2C_API_URL = os.getenv("MIKE_B2C_API_URL", "http://localhost:3000")


def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
    *,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> TracerProvider:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key or not host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set")

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider, metadata=metadata)
    return trace_provider


def parse_participant_metadata(participant: rtc.RemoteParticipant) -> dict:
    """Parse the participant metadata JSON which contains session params and JWT."""
    if not participant.metadata:
        return {"name": "aluno"}

    try:
        meta = json.loads(participant.metadata)
    except (json.JSONDecodeError, TypeError):
        # Fallback: try treating metadata as raw JWT (legacy format)
        try:
            decoded = jwt.decode(participant.metadata, options={"verify_signature": False})
            return {
                "name": decoded.get("given_name") or decoded.get("name", decoded.get("sub", "aluno")),
                "email": decoded.get("preferred_username"),
                "id": decoded.get("sub"),
            }
        except Exception:
            return {"name": "aluno"}

    result = {
        "topic": meta.get("topic", ""),
        "avatar": meta.get("avatar", "mike"),
        "lang": meta.get("lang", "en"),
        "roleplay": meta.get("roleplay", ""),
        "scenarioId": meta.get("scenarioId", ""),
        "level": meta.get("level", ""),
        "lesson": meta.get("lesson", ""),
    }

    # Extract user info from the JWT inside the metadata
    user_jwt = meta.get("jwt", "")
    if user_jwt:
        try:
            decoded = jwt.decode(user_jwt, options={"verify_signature": False})
            result["name"] = decoded.get("name") or decoded.get("given_name") or decoded.get("sub", "aluno")
            result["email"] = decoded.get("email") or decoded.get("preferred_username", "")
            result["id"] = decoded.get("userId") or decoded.get("sub", "")
        except Exception as e:
            logger.warning(f"Failed to decode JWT from metadata: {e}")
            result["name"] = "aluno"
    else:
        result["name"] = "aluno"

    return result


def fetch_agent_context(params: dict) -> dict | None:
    """Call the mike-b2c /api/agent-context endpoint to get the full instruction set."""
    try:
        query = {
            "topic": params.get("topic", ""),
            "avatar": params.get("avatar", "mike"),
            "lang": params.get("lang", "en"),
            "roleplay": "1" if params.get("roleplay") == "1" else "",
            "scenarioId": params.get("scenarioId", ""),
            "level": params.get("level", ""),
            "lesson": params.get("lesson", ""),
        }
        url = f"{MIKE_B2C_API_URL}/api/agent-context"
        logger.info(f"Fetching agent context from {url} with params: {query}")

        resp = requests.get(url, params=query, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Agent context received: voice={data.get('voice')}, "
                       f"teacherName={data.get('teacherName')}, "
                       f"instruction length={len(data.get('systemInstruction', ''))}")
            return data
        else:
            logger.error(f"Agent context API returned {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch agent context: {e}")
        return None


# Default fallback instruction when the API is unreachable
FALLBACK_INSTRUCTION = """
Você é o Professor Mike, um professor de inglês brasileiro experiente e muito paciente.
REGRA FUNDAMENTAL: Quando o usuário fala em PORTUGUÊS, responda em PORTUGUÊS primeiro,
depois convide para praticar em inglês. Quando o usuário fala em INGLÊS, responda só em inglês.
CORREÇÃO ATIVA: Você SEMPRE identifica e corrige erros de pronúncia, gramática ou vocabulário.
Estilo: claro, conciso, amigável; evite teoria longa; sempre feche com ação.
"""


class MikeAgent(Agent):
    def __init__(self, instructions: str, initial_message: str = "") -> None:
        super().__init__(instructions=instructions)
        self._initial_message = initial_message

    async def on_enter(self):
        if self._initial_message:
            self.session.generate_reply(instructions=self._initial_message)
        else:
            self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    params = parse_participant_metadata(participant)

    user_name = params.get("name", "aluno")
    user_id = params.get("id") or user_name
    user_email = params.get("email", "")

    logger.info(f"Participant joined: name={user_name}, topic={params.get('topic')}, "
                f"avatar={params.get('avatar')}, lang={params.get('lang')}, "
                f"roleplay={params.get('roleplay')}, scenarioId={params.get('scenarioId')}, "
                f"level={params.get('level')}, lesson={params.get('lesson')}")

    # Fetch full context from mike-b2c API
    context = fetch_agent_context(params)

    if context:
        system_instruction = context.get("systemInstruction", FALLBACK_INSTRUCTION)
        initial_message = context.get("initialMessage", "")
        voice = context.get("voice", "Charon")
        timing_prompts = context.get("timingPrompts", {})
        lesson_duration = context.get("lessonDurationSec", 300)
    else:
        # Fallback to simple agent if API unavailable
        system_instruction = f"Se apresente como Professor Mike e comece uma aula para {user_name}.\n{FALLBACK_INSTRUCTION}"
        initial_message = ""
        voice = "Charon" if params.get("avatar", "mike") != "nina" else "Zephyr"
        timing_prompts = {}
        lesson_duration = 300

    # Personalize with user name
    system_instruction = system_instruction.replace("{userName}", user_name)

    try:
        trace_provider = setup_langfuse(
            metadata={
                "langfuse.session.id": ctx.room.name,
                "langfuse.user.id": user_id,
                "user.email": user_email,
            }
        )

        async def flush_trace():
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_trace)
    except Exception as e:
        logger.warning(f"Langfuse setup failed (continuing without tracing): {e}")

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            voice=voice,
            temperature=0.8,
        )
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=MikeAgent(
            instructions=system_instruction,
            initial_message=initial_message,
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room=ctx.room,
    )

    # Schedule timing prompts (pronunciation warning, ending warning)
    async def send_timing_prompts():
        try:
            pron_warning = timing_prompts.get("pronunciationWarning", {})
            end_warning = timing_prompts.get("endingWarning", {})

            pron_at = pron_warning.get("atSecRemaining", 60)
            end_at = end_warning.get("atSecRemaining", 10)

            pron_delay = lesson_duration - pron_at
            end_delay = lesson_duration - end_at

            if pron_delay > 0 and pron_warning.get("message"):
                await asyncio.sleep(pron_delay)
                logger.info("Sending pronunciation warning")
                session.generate_reply(instructions=pron_warning["message"])

            remaining_wait = end_delay - pron_delay
            if remaining_wait > 0 and end_warning.get("message"):
                await asyncio.sleep(remaining_wait)
                logger.info("Sending ending warning")
                session.generate_reply(instructions=end_warning["message"])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Timing prompt error: {e}")

    asyncio.create_task(send_timing_prompts())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
