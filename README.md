# LangGraph + Google GenAI Conversation API

A production-ready FastAPI service that orchestrates **multi-step conversation workflows** using **LangGraph** and **Google Generative AI**. The service implements intelligent agent workflows with context validation, document retrieval, and streaming responses via Server-Sent Events (SSE).

## Overview

This project demonstrates a complete AI conversation system with:
- **LangGraph Workflows**: Multi-step agent execution with conditional routing and state management
- **Google GenAI Integration**: Powered by `google-genai` with streaming support
- **SSE Streaming**: Real-time responses streamed to clients
- **LangSmith Observability**: Full tracing and usage analytics
- **Modular Architecture**: Clean separation of concerns with services, routes, and workflows

## Features
- FastAPI app with LangGraph-based agent workflow orchestration
- Intelligent multi-node workflow: context validation → RAG retrieval → answer generation
- Conditional routing based on context and document availability
- SSE streaming for real-time response delivery
- Google GenAI (Gemini) model integration via `google-genai` SDK
- LangChain/LangSmith instrumentation for observability and token tracking
- CORS support for cross-origin requests
- Docker dev container with hot-reload capability

## Architecture

### Data Flow
```
User Request (/conversation)
    ↓
ConversationService
    ↓
AgentWorkFlow (LangGraph)
    ├── check_context_node: Validates if prompt has sufficient context
    ├─→ retrieve_rag_node: Searches and retrieves relevant documents
    ├─→ generate_answer_node: Produces response based on documents
    └─→ cannot_answer_node: Returns error if context/documents insufficient
    ↓
StreamingResponse (SSE)
    ↓
Client (real-time chunks)
```

### Components
- **API Layer** (`src/main.py`): FastAPI app with lifespan management, CORS middleware
- **Routes** (`src/routes/conversation.py`): `POST /conversation` endpoint accepting messages
- **Services** (`src/services/conversation_service.py`): Business logic orchestrating workflow
- **Workflows** (`src/workflows/agent_workflow.py`): LangGraph state machine with multi-step nodes
- **Dependencies** (`src/dependencies/app.py`): Settings, client initialization, middleware setup

## Requirements
- Python 3.12+
- Google API key with access to the chosen Vertex AI model
- (Optional) LangSmith credentials to record traces

## Configuration
Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your-google-api-key
MODEL_ID=gemini-1.5-flash

# CORS (use "*" for development, restrict in production)
BACKEND_CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Optional: LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=your-project-name
```

## Local Development
Install dependencies and run the API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### Test with cURL
```bash
curl -N \
	-H "Content-Type: application/json" \
	-d '{"message":"What is machine learning?"}' \
	http://localhost:8000/conversation
```

Response arrives as SSE chunks:
```
data: Machine learning...
data: is a subset...
data: of artificial...
```

## Docker

Build and run with hot-reload enabled:

```bash
# Build the image
docker build -t langgraph-google-genai -f Dockerfile.dev .

# Run with environment variables and volume mount
docker run --rm \
	--env-file .env \
	-p 8000:8000 \
	-v "$(pwd)/src:/app/src" \
	langgraph-google-genai
```

Or use the helper script:

```bash
bash ./up.sh
```

## API
- `POST /conversation` — Body: `{ "message": "Your prompt" }`. Returns `text/event-stream` where each `data:` line is a chunk.
- Streams conversational response based on user message through LangGraph agent workflow.

### Request / Response Example

**Request:**
```json
{
  "message": "What is the capital of France?"
}
```

**Response:** `text/event-stream` with headers:
```
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

Each line: `data: <chunk>\n\n`

**Example (cURL):**
```bash
curl -N \
	-H "Content-Type: application/json" \
	-d '{"message":"Explain quantum computing"}' \
	http://localhost:8000/conversation
```

**Example (Python):**
```python
import requests

with requests.post(
    "http://localhost:8000/conversation",
    json={"message": "Explain quantum computing"},
    stream=True,
    headers={"Accept": "text/event-stream"}
) as resp:
    for line in resp.iter_lines():
        if line:
            chunk = line.decode().replace("data: ", "")
            print(chunk, end="", flush=True)
```

## Project Structure
- `src/main.py` — FastAPI application entry point with lifespan management
- `src/dependencies/app.py` — FastAPI instance, settings, CORS middleware
- `src/routes/conversation.py` — `/conversation` endpoint definition
- `src/services/conversation_service.py` — Service orchestrating workflow execution
- `src/workflows/agent_workflow.py` — LangGraph state machine implementation
- `index.html` — Client for manual testing (optional)
- `Dockerfile.dev` — Development container with hot-reload

### Detailed Project Structure
```
src/
├── main.py                          # FastAPI application entry point
├── dependencies/
│   └── app.py                       # FastAPI instance, settings, middleware
├── routes/
│   └── conversation.py              # /conversation endpoint
├── services/
│   └── conversation_service.py      # Service orchestrating workflow execution
└── workflows/
    └── agent_workflow.py            # LangGraph state machine
```

## Workflow Details

The `AgentWorkFlow` class implements a 4-node LangGraph state machine:

1. **check_context_node**: Uses Gemini to validate if the prompt contains enough context
2. **retrieve_rag_node**: Simulates document retrieval (customize for your RAG backend)
3. **generate_answer_node**: Generates the final answer using retrieved documents
4. **cannot_answer_node**: Returns a polite refusal if context/documents are missing

Conditional edges route between nodes based on:
- `check_context_condition`: Routes to RAG or "cannot_answer"
- `retrieve_rag_condition`: Routes to generation or "cannot_answer"

## Configuration & Deployment Notes

### Development
- `BACKEND_CORS_ORIGINS` can be `["*"]` for local development
- `LANGCHAIN_TRACING_V2` enables LangSmith traces (requires valid credentials)
- Use `--reload` flag with uvicorn for hot-reloading

### Production
- Restrict `BACKEND_CORS_ORIGINS` to specific domains
- Set specific `allow_methods` and `allow_headers` in CORS middleware
- Use a production ASGI server (e.g., Gunicorn + Uvicorn)
- Ensure `GOOGLE_API_KEY` is securely managed (e.g., via secrets manager)
- Monitor LangSmith traces for performance and cost analysis

## License
This project is licensed under the terms of the LICENSE file in this repository.