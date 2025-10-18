# AI Agent Workflow System

A comprehensive platform for building, configuring, and executing dynamic AI agents with support for knowledge bases, tools, sub-agents, and more.

## Features

### Core Agent System
- **Dynamic Agent Configuration**: Toggle components on/off to build custom agent workflows
- **Multi-Provider Model Support**: OpenAI, Anthropic, Google, Azure, OpenRouter, and local models
- **Draft/Publish System**: Separate draft and published agent collections for safe deployments
- **Graph Compilation**: Automatic LangGraph generation based on configuration
- **Caching System**: Redis-based caching for compiled graphs and execution traces
- **Streaming Support**: Real-time execution with Server-Sent Events

### Knowledge Base System
- **Advanced Chunking**: Multiple chunking methods (recursive, token, sentence, semantic)
- **Multiple Embedding Models**: Support for various OpenAI embedding models
- **Query Enhancement**: Reranking and hybrid search capabilities
- **Playground Testing**: Interactive testing environment for retrieval configurations

### Tool System
- **Multiple Tool Types**: API, function, MCP, webhook, and LangChain built-in tools
- **LangChain Integration**: 50+ pre-configured LangChain tools (search, calculator, file operations)
- **Tool Management**: Easy addition/removal of tools with individual enable/disable controls
- **Custom Tools**: Support for custom Python implementations

### Advanced Features
- **Sub-Agent Architecture**: Hierarchical agent composition with trigger conditions
- **Execution Analytics**: Detailed execution statistics and performance monitoring
- **Model Registry**: Centralized model management with provider credentials
- **Validation System**: Comprehensive validation before publishing agents

## Architecture

The system follows a modular architecture with the following key components:

### Core Services
- **Agent Service**: Manages agent configurations and CRUD operations
- **Knowledge Base Service**: Handles vector storage and retrieval using Qdrant
- **Tool Service**: Executes various types of tools (API, function, MCP, webhook)
- **Execution Service**: Orchestrates agent execution and streaming

### Graph Compiler
- **Dynamic Graph Construction**: Builds LangGraph based on enabled components
- **Node Factory**: Creates specialized nodes for different capabilities
- **Conditional Edges**: Smart routing based on configuration
- **Sub-Graph Support**: Nested agent execution

### Caching Layer
- **Redis Integration**: Caches compiled graphs and metadata
- **Dependency Tracking**: Manages component relationships
- **Execution Traces**: Debugging and monitoring support

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (for services)
- OpenAI API key (or other provider keys)

### Option 1: Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-workflow
```

2. Set environment variables:
```bash
# Required
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional - for other providers
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
export COHERE_API_KEY="your_cohere_key"
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
```

3. Start all services:
```bash
docker-compose up -d
```

4. Access the API:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 2: Local Development Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start required services with Docker:
```bash
# MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:7.0

# Redis
docker run -d -p 6379:6379 --name redis redis:7.2-alpine

# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

3. Set environment variables:
```bash
# Database connections
export MONGODB_URL="mongodb://localhost:27017"
export MONGODB_DATABASE="agent_workflow"
export REDIS_URL="redis://localhost:6379"
export QDRANT_URL="http://localhost:6333"

# API Keys (at least one required)
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
export COHERE_API_KEY="your_cohere_key"

# Azure OpenAI (if using)
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"

# Application settings
export SECRET_KEY="your_secret_key_here"
export DEBUG="true"
export ENVIRONMENT="development"
```

4. Run the application:
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Using Start Scripts

**Linux/macOS:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Windows:**
```cmd
scripts\start.bat
```

### Verification

1. Check if all services are running:
```bash
# Check API health
curl http://localhost:8000/health

# Check MongoDB
docker exec agent_workflow_mongodb mongosh --eval "db.runCommand('ping')"

# Check Redis
docker exec agent_workflow_redis redis-cli ping

# Check Qdrant
curl http://localhost:6333/collections
```

2. Run the demo:
```bash
python examples/demo.py
```

## API Usage

### Creating an Agent

```python
import requests

# Create a new agent
response = requests.post("http://localhost:8000/api/agents/", json={
    "name": "Customer Support Agent",
    "description": "Handles customer inquiries",
    "user_id": "user123"
})

agent_id = response.json()["id"]
```

### Configuring Agent Components

```python
# Enable knowledge base
requests.patch(f"http://localhost:8000/api/agents/{agent_id}/toggle", json={
    "path": "knowledge_bases.enabled",
    "value": True
})

# Add a knowledge base
requests.post(f"http://localhost:8000/api/agents/{agent_id}/knowledge-bases", 
              params={"kb_id": "str", "priority": 1})

# Enable tools
requests.patch(f"http://localhost:8000/api/agents/{agent_id}/toggle", json={
    "path": "tools.enabled",
    "value": True
})
```

### Publishing and Executing

```python
# Publish the agent
requests.post(f"http://localhost:8000/api/agents/{agent_id}/publish")

# Execute the agent
response = requests.post("http://localhost:8000/api/execution/", json={
    "agent_id": agent_id,
    "input": "Hello, I need help with my order",
    "stream": False
})

print(response.json()["output"])
```

### Streaming Execution

```python
# Stream execution
response = requests.post("http://localhost:8000/api/execution/stream", json={
    "agent_id": agent_id,
    "input": "Tell me about your products",
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## Configuration Examples

### Basic Customer Support Agent

```json
{
  "llm": {
    "enabled": true,
    "model": "gpt-4",
    "system_prompt": "You are a helpful customer support agent."
  },
  "knowledge_bases": {
    "enabled": true,
    "items": [
      {
        "id": "faq_kb",
        "enabled": true,
        "priority": 1
      }
    ]
  },
  "tools": {
    "enabled": true,
    "items": [
      {
        "id": "crm_lookup",
        "enabled": true
      }
    ]
  },
  "memory": {
    "enabled": true
  }
}
```

### Advanced Agent with Sub-Agents

```json
{
  "llm": {
    "enabled": true,
    "model": "gpt-4"
  },
  "knowledge_bases": {
    "enabled": true
  },
  "tools": {
    "enabled": true
  },
  "sub_agents": {
    "enabled": true,
    "items": [
      {
        "id": "refund_specialist",
        "enabled": true,
        "trigger_condition": "user_intent == 'refund'"
      },
      {
        "id": "technical_support",
        "enabled": true,
        "trigger_condition": "contains(message, 'technical')"
      }
    ]
  }
}
```

## Development

### Project Structure

```
app/
├── api/                    # API routes and endpoints
│   └── routes/
│       ├── agents.py       # Agent management endpoints
│       ├── knowledge_bases.py  # Knowledge base endpoints
│       ├── tools.py        # Tool management endpoints
│       └── execution.py    # Execution endpoints
├── core/                   # Core services and utilities
│   ├── cache.py           # Redis caching
│   └── graph_compiler.py  # LangGraph compilation
├── models/                 # Pydantic models
│   ├── agent.py           # Agent models
│   ├── knowledge_base.py  # Knowledge base models
│   ├── tool.py            # Tool models
│   └── execution.py       # Execution models
├── services/               # Business logic services
│   ├── agent_service.py   # Agent operations
│   ├── knowledge_base_service.py  # Knowledge base operations
│   ├── tool_service.py    # Tool operations
│   └── execution_service.py  # Execution operations
├── database.py            # Database connection and setup
└── main.py                # FastAPI application
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

### Code Quality

```bash
# Format code
black app/
isort app/

# Type checking
mypy app/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
