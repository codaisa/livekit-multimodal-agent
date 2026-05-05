"""
---
title: Onboarding Voice Agent
category: education
tags: [education, onboarding, livekit, voice-agent]
---
"""

import logging
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, room_io
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import google

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("onboarding-agent")
logger.setLevel(logging.DEBUG)

VERSION = "0.2.0"

SESSION_DURATION_SEC = 60

logger.info(f"🚀 onboarding-agent v{VERSION} loaded")

MIKE_B2C_URL = os.getenv("MIKE_B2C_URL", "https://app.falamike.com")
MIKE_INTERNAL_API_KEY = os.getenv("MIKE_INTERNAL_API_KEY", "uma-chave-secreta-qualquer-aqui")

# Fallback hard-coded usado apenas se o backend não enviar `agentContext`
# na metadata (modo console ou erro ao buscar o prompt no banco).
FALLBACK_ONBOARDING_INSTRUCTION = f"""
Você é o Mike, um professor de inglês super simpático e acolhedor.
Você está conhecendo um novo aluno pela primeira vez. Sua missão é ter uma conversa leve e natural
para descobrir três coisas sobre ele:

1. **Nome** — Pergunte como ele se chama (ou como gosta de ser chamado).
2. **Interesses** — Descubra o que ele gosta de fazer, seus hobbies, o que o anima.
3. **Bloqueios com inglês** — Entenda o que dificulta ou impede ele de aprender/falar inglês.

REGRAS DE TEMPO:
- Esta conversa deve durar EXATAMENTE {SESSION_DURATION_SEC} segundos (1 minuto).
- Você tem cerca de 20 segundos para cada pergunta (pergunta + resposta do aluno + seu comentário).
- Se o aluno estiver sendo breve demais, explore mais a resposta dele com follow-ups curtos.
- Se o aluno estiver falando demais, gentilmente reconheça e avance para a próxima pergunta.

REGRAS GERAIS:
- Fale em PORTUGUÊS (o aluno ainda não pratica inglês nesta etapa).
- Seja natural, NÃO faça as 3 perguntas de uma vez. Conduza como uma conversa real.
- Depois de cada resposta, faça um breve comentário positivo antes de prosseguir.
- Seja breve nas suas falas — máximo 2-3 frases por vez.
- NÃO corrija inglês nesta etapa — isso é só onboarding.

REGRA OBRIGATÓRIA DE ENCERRAMENTO:
- Quando você já tiver coletado as 3 informações (nome, interesses e bloqueios), ou quando sentir que o tempo está acabando, você DEVE encerrar a conversa.
- A frase de encerramento DEVE OBRIGATORIAMENTE conter as palavras exatas "Bora começar" — isso é um gatilho técnico que o sistema usa para detectar o fim da conversa.
- Exemplo: "Massa! Agora a gente vai montar um plano perfeito pra você. Bora começar?"
- NUNCA termine a conversa sem dizer "Bora começar". Isso é CRÍTICO para o funcionamento do sistema.
"""


def prewarm(proc: JobProcess):
    logger.info("[PREWARM] Pre-importing heavy modules...")
    import google.genai  # noqa: F401
    from livekit.plugins import google as _google_plugin  # noqa: F401
    import aiohttp  # noqa: F401
    logger.info("[PREWARM] Process warmed up and ready!")


class OnboardingAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)

    async def on_enter(self):
        logger.info("[ONBOARDING AGENT] Triggering initial greeting")
        await self.session.generate_reply()


async def entrypoint(ctx: JobContext):

    # ── Parse metadata ──
    meta = {}
    if ctx.job.metadata:
        try:
            meta = json.loads(ctx.job.metadata)
            logger.info(f"[METADATA] Parsed keys from job: {list(meta.keys())}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[METADATA] Failed to parse job metadata: {e}")

    user = meta.get("user", {})
    user_id = user.get("id")
    user_name = user.get("name", "aluno")
    voice = meta.get("voice", "Charon")
    session_id = meta.get("sessionId")  # onboarding session ID from B2C
    tenant_id = meta.get("tenantId")
    agent_context = meta.get("agentContext")

    # Allow console mode (no metadata) for local testing
    is_console = not ctx.job.metadata
    if not is_console and not user_id:
        logger.warning("[ENTRYPOINT] Rejecting session — no user_id")
        ctx.shutdown("unauthorized")
        return

    # Build system instruction: prefer the prompt sent by the backend
    # (resolved from aiPrompt by tenant or global fallback). Cair no
    # template hard-coded apenas quando vier vazio.
    if agent_context:
        system_instruction = agent_context
        logger.info(
            f"[ENTRYPOINT] agentContext received ({len(system_instruction)} chars, tenantId={tenant_id})"
        )
    else:
        system_instruction = FALLBACK_ONBOARDING_INSTRUCTION
        logger.warning("[ENTRYPOINT] No agentContext in metadata — using FALLBACK")

    logger.info(f"[ENTRYPOINT] Onboarding for user={user_name} (id={user_id}), voice={voice}, sessionId={session_id}")

    # ── Create agent session ──
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice=voice,
            temperature=0.8,
        )
    )

    # ── Send transcript on session end ──
    async def on_session_end(reason: str) -> None:
        logger.info(f"[TRANSCRIPT] Session ended, reason: {reason}")
        try:
            report = ctx.make_session_report()
            report_dict = report.to_dict()

            if MIKE_INTERNAL_API_KEY and session_id:
                url = f"{MIKE_B2C_URL}/api/onboarding/session/{session_id}/transcript"
                payload = {
                    "chatHistory": report_dict.get("chat_history", []),
                    "livekitRoomName": ctx.room.name,
                    "livekitJobId": ctx.job.id,
                    "durationSeconds": SESSION_DURATION_SEC,
                }
                async with aiohttp.ClientSession() as http:
                    async with http.post(
                        url,
                        json=payload,
                        headers={"X-Agent-Key": MIKE_INTERNAL_API_KEY},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        body = await resp.json()
                        if resp.status == 200:
                            logger.info(f"[TRANSCRIPT] Sent to mike-b2c: {body}")
                        else:
                            logger.error(f"[TRANSCRIPT] mike-b2c responded {resp.status}: {body}")
            else:
                if not session_id:
                    logger.warning("[TRANSCRIPT] No sessionId in metadata, skipping API send")
                elif not MIKE_INTERNAL_API_KEY:
                    logger.warning("[TRANSCRIPT] MIKE_INTERNAL_API_KEY not set, skipping API send")
        except Exception as e:
            logger.error(f"[TRANSCRIPT] Failed to save: {e}")

    ctx.add_shutdown_callback(on_session_end)

    # ── Hard timer: force shutdown after SESSION_DURATION_SEC ──
    async def session_timer():
        # Wait until 50s to inject a "wrap up" hint
        await asyncio.sleep(SESSION_DURATION_SEC - 10)
        logger.info("[TIMER] 10 seconds remaining — signaling agent to wrap up")
        session.say("Bom, nosso tempo tá acabando! Foi ótimo te conhecer. Vou preparar tudo pra gente começar!")

        # Wait the final 10 seconds, then force disconnect
        await asyncio.sleep(10)
        logger.info(f"[TIMER] {SESSION_DURATION_SEC}s reached — ending session")
        ctx.shutdown("session_time_limit")

    # ── Connect and start ──
    logger.info("[ENTRYPOINT] Connecting to room...")
    await ctx.connect()
    logger.info("[ENTRYPOINT] Connected! Starting onboarding session...")

    await session.start(
        agent=OnboardingAgent(instructions=system_instruction),
        room=ctx.room,
    )
    logger.info("[ENTRYPOINT] Onboarding agent session started!")

    # Start the hard timer after session is running
    asyncio.create_task(session_timer())


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=2,
            agent_name="onboarding-agent",
            port=8082,
        )
    )