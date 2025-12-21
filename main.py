"""
---
title: Pipeline Translator Agent
category: translation
tags: [translation, multilingual, french, elevenlabs, direct-translation]
difficulty: intermediate
description: Simple translation pipeline that converts English speech to French
demonstrates:
  - Direct language translation workflow
  - Multilingual TTS configuration with ElevenLabs
  - Simple translation-focused agent instructions
  - Clean input-to-output translation pipeline
  - Voice-to-voice translation system
---
"""

import logging
import os
import base64
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

logger = logging.getLogger("pipeline-translator")
logger.setLevel(logging.INFO)

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

class SimpleAgent(Agent):
    def __init__(self, user_name: str = "aluno") -> None:
        super().__init__(
            instructions=f"""
                Se apresente como: Olá {user_name}! Eu sou o Professor Mike, seu professor de inglês. Vou te ajudar a praticar conversação de forma divertida e educativa. Pergunte como ele gostaria de começar nossa aula de conversação. O nome do seu aluno é {user_name}. Use o nome dele para tornar a conversa mais pessoal.
            
                Você é Professor Mike, um professor de inglês brasileiro experiente e muito paciente. \
                REGRA FUNDAMENTAL: Quando o usuário fala em PORTUGUÊS, você DEVE responder em PORTUGUÊS primeiro, \
                depois convidar gentilmente para praticar em inglês. Quando o usuário fala em INGLÊS, responda só em inglês. \
                CORREÇÃO ATIVA: Você SEMPRE identifica e corrige erros de pronúncia, gramática ou vocabulário. \
                Não deixe nenhum erro passar despercebido - você é um professor ativo e atento. \
                CORREÇÕES SEMPRE EM PORTUGUÊS: Quando corrigir, SEMPRE faça em português brasileiro primeiro: \
                "Ótimo! Mas deixa eu te ajudar: 'palavra' se pronuncia assim /pronúncia/, tenta de novo. Great! Now continue in English." \
                Use símbolos fonéticos quando necessário para explicar pronúncia. \
                Você é gentil, encorajador mas rigoroso na correção de erros. \
                Você fala de forma clara e pausada para facilitar o entendimento. \
                Suas respostas são concisas e focadas no aprendizado e correção. \
                Você sempre elogia o progresso do usuário mas nunca deixa erros passarem. \
                REGRA DE OURO: Todas as correções começam em português brasileiro, depois continua em inglês.
                Estilo: claro, conciso, amigável; evite teoria longa; sempre feche com ação (pergunta/drill/próximo passo).
            """,
        )
    
    async def on_enter(self):
        self.session.generate_reply()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    def get_user_info(participant: rtc.RemoteParticipant) -> dict:
        if not participant.metadata:
            return {"name": "aluno"}
        
        try:
            decoded = jwt.decode(participant.metadata, options={"verify_signature": False})
            return {
                "name": decoded.get("given_name") or decoded.get("name", decoded.get("sub", "aluno")),
                "email": decoded.get("preferred_username"),
                "id": decoded.get("sub"),
            }
        except Exception as e:
            logger.warning(f"Erro ao decodificar JWT: {e}")
            return {"name": "aluno"}
    
    participant = await ctx.wait_for_participant()
    user_info = get_user_info(participant)
   
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
            "langfuse.user.id": user_info.get("id") or user_info["name"],
            "user.email": user_info.get("email") or "",
        }
    )

    # (optional) add a shutdown callback to flush the trace before process exit
    async def flush_trace():
        trace_provider.force_flush()

    ctx.add_shutdown_callback(flush_trace)
    
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


    await session.start(
        agent=SimpleAgent(user_name=user_info["name"]),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))