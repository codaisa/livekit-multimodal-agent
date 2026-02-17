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
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, RoomInputOptions, metrics
from livekit.agents.voice import Agent, AgentSession, MetricsCollectedEvent
from livekit.plugins import openai, silero, deepgram, elevenlabs, google, noise_cancellation
from livekit.agents.telemetry import set_tracer_provider
from livekit import rtc

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

load_dotenv('.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("mike-voice-agent")
logger.setLevel(logging.DEBUG)


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
    """Parse the participant metadata JSON which contains user, agentContext."""
    logger.info(f"[METADATA] Participant identity: {participant.identity}")

    if not participant.metadata:
        logger.warning("[METADATA] No metadata found on participant!")
        return {}

    try:
        meta = json.loads(participant.metadata)
        logger.info(f"[METADATA] Parsed metadata keys: {list(meta.keys())}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"[METADATA] Failed to parse metadata as JSON: {e}")
        return {}

    result = {}

    # Extract user info (set by livekit.js from the JWT)
    user = meta.get("user", {})
    result["name"] = user.get("name", "aluno")
    result["email"] = user.get("email", "")
    result["id"] = user.get("id", "")
    logger.info(f"[METADATA] User: name={result['name']}, email={result['email']}, id={result['id']}")

    # Extract agentContext (the full context built by the frontend)
    agent_context = meta.get("agentContext")
    if agent_context:
        result["agentContext"] = agent_context
        logger.info(f"[METADATA] agentContext found with keys: {list(agent_context.keys())}")
    else:
        logger.warning("[METADATA] No agentContext in metadata!")

    return result


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

    parsed = parse_participant_metadata(participant)

    user_name = parsed.get("name", "aluno")
    user_id = parsed.get("id") or user_name
    user_email = parsed.get("email", "")
    context = parsed.get("agentContext")


    if context:
        system_instruction = context.get("systemInstruction", FALLBACK_INSTRUCTION)
        initial_message = context.get("initialMessage", "")
        voice = context.get("voice", "Charon")
        timing_prompts = context.get("timingPrompts", {})
        lesson_duration = context.get("lessonDurationSec", 300)
        logger.info(f"[ENTRYPOINT] Using frontend context: voice={voice}, instruction={len(system_instruction)} chars")
    else:
        system_instruction = f"Se apresente como Professor Mike e comece uma aula para {user_name}.\n{FALLBACK_INSTRUCTION}"
        initial_message = ""
        voice = "Charon"
        timing_prompts = {}
        lesson_duration = 300
        logger.warning(f"[ENTRYPOINT] No agentContext in metadata, using FALLBACK")

    # Personalize with user name
    system_instruction = system_instruction.replace("{userName}", user_name)
    system_instruction += f"\n\nO NOME DO ALUNO É: {user_name}. REGRA OBRIGATÓRIA: Ao iniciar a conversa, SEMPRE cumprimente o aluno pelo nome (ex: \"Olá, {user_name}!\"). Use o nome dele ao longo da aula também."

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
        logger.info("[ENTRYPOINT] Langfuse tracing initialized")
    except Exception as e:
        logger.warning(f"[ENTRYPOINT] Langfuse setup failed (continuing without tracing): {e}")

    logger.info(f"[ENTRYPOINT] Creating AgentSession with model=gemini-2.5-flash-native-audio-preview-09-2025, voice={voice}")
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

    logger.info(f"[ENTRYPOINT] Starting agent session...")
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
    logger.info(f"[ENTRYPOINT] Agent session started successfully!")

    # Schedule timing prompts (pronunciation warning, ending warning)
    async def send_timing_prompts():
        try:
            pron_warning = timing_prompts.get("pronunciationWarning", {})
            end_warning = timing_prompts.get("endingWarning", {})

            pron_at = pron_warning.get("atSecRemaining", 60)
            end_at = end_warning.get("atSecRemaining", 10)

            pron_delay = lesson_duration - pron_at
            end_delay = lesson_duration - end_at

            logger.info(f"[TIMING] Scheduled: pronunciation warning at {pron_delay}s, ending warning at {end_delay}s (lesson={lesson_duration}s)")

            if pron_delay > 0 and pron_warning.get("message"):
                await asyncio.sleep(pron_delay)
                logger.info("[TIMING] Sending pronunciation warning NOW")
                session.generate_reply(instructions=pron_warning["message"])

            remaining_wait = end_delay - pron_delay
            if remaining_wait > 0 and end_warning.get("message"):
                await asyncio.sleep(remaining_wait)
                logger.info("[TIMING] Sending ending warning NOW")
                session.generate_reply(instructions=end_warning["message"])

            logger.info("[TIMING] All timing prompts sent")
        except asyncio.CancelledError:
            logger.info("[TIMING] Timing prompts cancelled (session ended)")
        except Exception as e:
            logger.error(f"[TIMING] Error: {type(e).__name__}: {e}")

    asyncio.create_task(send_timing_prompts())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
