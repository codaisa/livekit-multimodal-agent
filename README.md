# LiveKit English Teacher Agent

Um agente de voz inteligente que atua como professor de inglês interativo, utilizando LiveKit para comunicação em tempo real e Google Gemini para processamento de linguagem natural.

## Descrição

Professor Mike é um assistente de conversação alimentado por IA que ajuda estudantes brasileiros a praticar inglês. O agente:

- Responde em português quando o usuário fala português, e convida gentilmente para praticar em inglês
- Corrige ativamente erros de pronúncia, gramática e vocabulário
- Fornece feedback em português brasileiro para melhor compreensão
- Utiliza símbolos fonéticos para explicar pronúncia
- Mantém conversas naturais e educativas em tempo real

## Tecnologias

- **LiveKit**: Comunicação em tempo real (WebRTC)
- **Google Gemini 2.5 Flash**: Modelo de IA com suporte a áudio nativo
- **Langfuse**: Observabilidade e telemetria
- **Noise Cancellation**: Cancelamento de ruído (BVC)
- **Python 3.13**: Runtime

## Pré-requisitos

- Python 3.13+
- Docker (opcional)
- Conta LiveKit
- Chaves de API:
  - Google AI (Gemini)
  - Langfuse
  - LiveKit

## Instalação

### Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas credenciais
```

### Docker

```bash
# Build da imagem
docker build -t livekit-english-teacher .

# Executar container
docker run --env-file .env livekit-english-teacher
```

## Configuração

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://seu-servidor.livekit.cloud
LIVEKIT_API_KEY=sua_api_key
LIVEKIT_API_SECRET=seu_api_secret

# Google AI (Gemini)
GOOGLE_API_KEY=sua_google_api_key

# Langfuse (Observabilidade)
LANGFUSE_PUBLIC_KEY=sua_langfuse_public_key
LANGFUSE_SECRET_KEY=sua_langfuse_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Opcional: Outras APIs
DEEPGRAM_API_KEY=sua_deepgram_key
OPENAI_API_KEY=sua_openai_key
ELEVEN_API_KEY=sua_elevenlabs_key
```

## Uso

### Desenvolvimento Local

```bash
# Iniciar o agente
python main.py start

# Download de arquivos/modelos necessários
python main.py download-files
```

### Docker

```bash
# Executar em foreground
docker run --env-file .env --name english-teacher livekit-english-teacher

# Executar em background
docker run -d --env-file .env --name english-teacher livekit-english-teacher

# Ver logs
docker logs -f english-teacher

# Parar container
docker stop english-teacher
```

## Estrutura do Projeto

```
livekit-stt-agent/
├── main.py              # Código principal do agente
├── requirements.txt     # Dependências Python
├── Dockerfile          # Configuração Docker
├── livekit.toml        # Configuração LiveKit
├── .env                # Variáveis de ambiente (não commitado)
├── .gitignore          # Arquivos ignorados pelo Git
└── .dockerignore       # Arquivos ignorados pelo Docker
```

## Funcionalidades

### SimpleAgent (Professor Mike)

- **Detecção de idioma**: Identifica se o usuário está falando em português ou inglês
- **Correção ativa**: Corrige todos os erros de pronúncia, gramática e vocabulário
- **Feedback em português**: Explicações sempre em português brasileiro
- **Encorajamento**: Elogia o progresso mas mantém rigor pedagógico
- **Conversação natural**: Mantém diálogo fluido e educativo

### Métricas e Observabilidade

O agente coleta métricas através do Langfuse, incluindo:
- Latência de resposta
- Qualidade das interações
- Uso de tokens
- Sessões por sala

## Deploy

### LiveKit Cloud

1. Configure seu projeto no [LiveKit Cloud](https://cloud.livekit.io)
2. Obtenha suas credenciais (URL, API Key, Secret)
3. Atualize o `livekit.toml` com seu subdomain e agent ID
4. Deploy usando Docker ou Python direto

### Self-hosted

Consulte a [documentação do LiveKit](https://docs.livekit.io/agents/ops/deployment/builds/) para mais informações sobre deploy self-hosted.

## Desenvolvimento

### Modificar o comportamento do agente

Edite as instruções em `main.py:64-81` para ajustar a personalidade e regras do Professor Mike.

### Trocar o modelo de IA

Altere o `RealtimeModel` em `main.py:101-105` para usar outro modelo ou configurações diferentes.

### Adicionar plugins

Instale plugins adicionais via `requirements.txt` e importe em `main.py:24`.

## Troubleshooting

### Erro de autenticação LiveKit

Verifique se `LIVEKIT_URL`, `LIVEKIT_API_KEY` e `LIVEKIT_API_SECRET` estão corretos no `.env`.

### Modelo Gemini não responde

Confirme que `GOOGLE_API_KEY` é válida e tem acesso ao modelo `gemini-2.5-flash-native-audio-preview-09-2025`.

### Langfuse não está enviando dados

Verifique as credenciais `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` e `LANGFUSE_HOST`.

## Licença

Este projeto é de uso educacional.

## Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

## Recursos Adicionais

- [Documentação LiveKit Agents](https://docs.livekit.io/agents/)
- [Google Gemini API](https://ai.google.dev/)
- [Langfuse Docs](https://langfuse.com/docs)
