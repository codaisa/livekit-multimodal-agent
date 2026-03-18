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
import aiohttp
from datetime import datetime, timezone
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, RoomInputOptions, metrics
from livekit.agents.voice import Agent, AgentSession, MetricsCollectedEvent
from livekit.plugins import openai, silero, deepgram, elevenlabs, google
from livekit.agents.telemetry import set_tracer_provider
from livekit import rtc

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

load_dotenv('.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("mike-voice-agent")
logger.setLevel(logging.DEBUG)


# ═══════════════════════════════════════════════════════════════
# PREWARM — roda uma vez por processo idle, ANTES de qualquer job
# ═══════════════════════════════════════════════════════════════
def prewarm(proc: JobProcess):
    """
    Pré-carrega módulos pesados no processo idle para que,
    quando um job chegar, não haja custo de import.
    """
    logger.info("[PREWARM] Pre-importing heavy modules...")

    # Força import dos módulos que demoram mais no primeiro load
    import google.genai  # noqa: F401
    from livekit.plugins import google as _google_plugin  # noqa: F401
    import aiohttp  # noqa: F401
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: F401
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: F401

    logger.info("[PREWARM] Process warmed up and ready!")


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


MIKE_B2C_URL = os.getenv("MIKE_B2C_URL", "https://app.falamike.com")
MIKE_INTERNAL_API_KEY = os.getenv("MIKE_INTERNAL_API_KEY", "uma-chave-secreta-qualquer-aqui")

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
        logger.info("[MIKE AGENT] Triggering initial reply")
        await self.session.generate_reply()


async def entrypoint(ctx: JobContext):

    # ── Parse metadata from job dispatch (available immediately) ──
    meta = {}
    if ctx.job.metadata:
        try:
            meta = json.loads(ctx.job.metadata)
            logger.info(f"[METADATA] Parsed keys from job: {list(meta.keys())}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[METADATA] Failed to parse job metadata: {e}")

    user = meta.get("user", {})
    user_name = user.get("name", "aluno")
    user_id = user.get("id")
    user_email = user.get("email", "")
    logger.info(f"[METADATA] user={user}")

    # Reject sessions without valid metadata (e.g. SIP scanners)
    if not user_id or not meta.get("agentContext"):
        logger.warning(f"[ENTRYPOINT] Rejecting session — no user_id or agentContext")
        ctx.shutdown("unauthorized")
        return

    agent_context = meta.get("agentContext", "")
    lesson_duration = meta.get("lessonDurationSec", 300)
    voice = meta.get("voice", "Charon")
    language = meta.get("language")
    lesson_id = meta.get("lessonId")
    lesson_attempt_id = meta.get("lessonAttemptId")
    roleplay_scenario_id = meta.get("roleplayScenarioId")
    conversation_session_id = meta.get("conversationSessionId")

    # ── Build system instruction ────────────────────────────
    if agent_context:
        system_instruction = agent_context
        logger.info(f"[ENTRYPOINT] agentContext received ({len(system_instruction)} chars)")
    else:
        system_instruction = FALLBACK_INSTRUCTION
        logger.warning("[ENTRYPOINT] No agentContext — using FALLBACK")

    logger.info(f"[ENTRYPOINT] user={user_name}, duration={lesson_duration}s, voice={voice}")

    # ── Langfuse tracing ────────────────────────────────────
    try:
        trace_provider = setup_langfuse(
            metadata={
                "langfuse.session.id": ctx.job.room.name,
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

    # ── Create agent session BEFORE connecting ──────────────
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice=voice,
            temperature=0.8,
        )
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    # ── Save transcript on session end ──────────────────────
    async def on_session_end(reason: str) -> None:
        logger.info(f"[TRANSCRIPT] Session ended, reason: {reason}")
        try:
            report = ctx.make_session_report()
            report_dict = report.to_dict()

            # Send to mike-b2c API
            if MIKE_INTERNAL_API_KEY and user_id:
                url = f"{MIKE_B2C_URL}/api/conversation/session"
                payload = {
                    "report": report_dict,
                    "userId": user_id,
                    "roomName": ctx.room.name,
                    "language": language,
                    "lessonAttemptId": lesson_attempt_id,
                    "roleplayScenarioId": roleplay_scenario_id,
                    "conversationSessionId": conversation_session_id,
                }
                async with aiohttp.ClientSession() as http:
                    async with http.post(
                        url,
                        json=payload,
                        headers={"X-Api-Key": MIKE_INTERNAL_API_KEY},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        body = await resp.json()
                        if resp.status == 200:
                            logger.info(f"[TRANSCRIPT] Sent to mike-b2c: {body}")
                        else:
                            logger.error(f"[TRANSCRIPT] mike-b2c responded {resp.status}: {body}")
            else:
                if not MIKE_INTERNAL_API_KEY:
                    logger.warning("[TRANSCRIPT] MIKE_INTERNAL_API_KEY not set, skipping API send")
                elif not user_id:
                    logger.warning("[TRANSCRIPT] No user_id in metadata, skipping API send")
        except Exception as e:
            logger.error(f"[TRANSCRIPT] Failed to save: {e}")

    ctx.add_shutdown_callback(on_session_end)

    # ── Connect and start immediately ───────────────────────
    logger.info("[ENTRYPOINT] Connecting to room...")
    await ctx.connect()
    logger.info("[ENTRYPOINT] Connected! Starting agent session...")

    await session.start(
        agent=MikeAgent(instructions=system_instruction),
        room=ctx.room,
    )
    logger.info("[ENTRYPOINT] Agent session started successfully!")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=3,
            agent_name="mike-agent",
        )
    )