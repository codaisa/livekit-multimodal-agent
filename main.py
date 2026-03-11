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


FALLBACK_INSTRUCTION = """
Você é o Professor Mike, um professor de inglês brasileiro experiente e muito paciente.
REGRA FUNDAMENTAL: Quando o usuário fala em PORTUGUÊS, responda em PORTUGUÊS primeiro,
depois convide para praticar em inglês. Quando o usuário fala em INGLÊS, responda só em inglês.
CORREÇÃO ATIVA: Você SEMPRE identifica e corrige erros de pronúncia, gramática ou vocabulário.
Estilo: claro, conciso, amigável; evite teoria longa; sempre feche com ação.
"""


class MikeAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)

    async def on_enter(self):
        await asyncio.sleep(1)
        logger.info("[MIKE AGENT] Triggering initial reply")
        await self.session.generate_reply()


async def entrypoint(ctx: JobContext):

    await ctx.connect()
    participant = await ctx.wait_for_participant()

    logger.info(f"[ENTRYPOINT] Participant identity: {participant.identity}")

    # ── Parse metadata ──────────────────────────────────────
    meta = {}
    if participant.metadata:
        try:
            meta = json.loads(participant.metadata)
            logger.info(f"[METADATA] Parsed keys: {list(meta.keys())}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[METADATA] Failed to parse: {e}")

    user = meta.get("user", {})
    user_name = user.get("name", "aluno")
    user_id = user.get("id") or user_name
    user_email = user.get("email", "")

    agent_context = meta.get("agentContext", "")
    lesson_duration = meta.get("lessonDurationSec", 300)

    # ── Build system instruction ────────────────────────────
    if agent_context:
        system_instruction = agent_context
        logger.info(f"[ENTRYPOINT] agentContext received ({len(system_instruction)} chars)")
    else:
        system_instruction = FALLBACK_INSTRUCTION
        logger.warning("[ENTRYPOINT] No agentContext — using FALLBACK")

    logger.info(f"[ENTRYPOINT] user={user_name}, duration={lesson_duration}s")

    # ── Langfuse tracing ────────────────────────────────────
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

    # ── Create and start agent session ──────────────────────
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            voice="Charon",
            temperature=0.8,
        )
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    logger.info("[ENTRYPOINT] Starting agent session...")
    await session.start(
        agent=MikeAgent(instructions=system_instruction),
        room=ctx.room,
    )
    logger.info("[ENTRYPOINT] Agent session started successfully!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
