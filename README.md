# Unified AI API

A modular REST API supporting multiple AI providers (Ollama, OpenAI, Gemini, Hugging Face) through a single unified interface.

## Quick Start

### 1. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Services

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or run locally
pip install -r requirements.txt
python unified_ai_api.py
```

### 3. Access API

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## Usage Examples

### Generate Text

```bash
# Using default provider (Ollama)
curl -X POST "http://localhost:8000/api/v1/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "stream": false
  }'

# Using specific provider
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "provider": "openai",
    "model": "gpt-4",
    "stream": false
  }'
```

### Chat Completion

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "provider": "ollama",
    "stream": false
  }'
```

### Generate with Images

```bash
curl -X POST "http://localhost:8000/api/v1/generate/upload" \
  -F "prompt=What is in this image?" \
  -F "files=@image.jpg" \
  -F "provider=ollama" \
  -F "model=llava"
```

### List Models

```bash
# All providers
curl "http://localhost:8000/api/v1/models"

# Specific provider
curl "http://localhost:8000/api/v1/models/ollama"
```

### Generate Embeddings

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "provider": "ollama"
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API authentication key | None |
| `PORT` | Server port | 8000 |
| `MAX_FILE_SIZE` | Max upload size (bytes) | 10485760 |
| `OLLAMA_ENABLED` | Enable Ollama | true |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OPENAI_ENABLED` | Enable OpenAI | false |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GEMINI_ENABLED` | Enable Gemini | false |
| `GEMINI_API_KEY` | Gemini API key | - |

## Provider Support

- ✅ **Ollama** - Local models
- ✅ **OpenAI** - GPT-3.5, GPT-4, etc.
- ✅ **Google Gemini** - Gemini Pro, etc.
- ✅ **Hugging Face** - Inference API
- 🔌 **Extensible** - Easy to add more

## Features

- 🔄 **Multi-Provider** - Use any AI provider through one API
- 🔀 **Auto-Routing** - Automatic provider selection
- 📊 **Streaming** - Real-time response streaming
- 🖼️ **Multimodal** - Image support where available
- 🔐 **Authentication** - Optional API key protection
- 📝 **OpenAPI Docs** - Auto-generated documentation
- 🐳 **Docker Ready** - Easy deployment
- ⚡ **Fast** - Async/await throughout

## Management Commands

```bash
make start      # Start services
make stop       # Stop services
make logs       # View logs
make restart    # Restart services
make clean      # Clean up
make shell      # Open container shell
```

## Health Check

```bash
curl http://localhost:8000/api/v1health
```

Returns provider availability status.

## License

MIT
