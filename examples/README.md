# LLMFS Examples

Copy this folder anywhere, install the requirements, and run.

## Quick start

```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama serve
```

Then:

```bash
# Scripted demo — proves LLMFS works end-to-end (5 automated steps)
python ollama_demo.py

# Interactive chat — long-term memory across sessions
python ollama_chat.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A model that supports tool/function calls, e.g.:
  - `llama3.2` (default)
  - `mistral`
  - `qwen2.5`

## Options

```bash
python ollama_demo.py --model mistral
python ollama_demo.py --model llama3.2 --store /tmp/my_store

python ollama_chat.py --model mistral
python ollama_chat.py --store ~/.my_llmfs   # persists across runs
```

## Chat slash commands

| Command | Description |
|---------|-------------|
| `/list` | Show all stored memories |
| `/search <query>` | Semantic search |
| `/forget <path>` | Delete a memory |
| `/status` | Storage stats |
| `/clear` | Wipe all memories |
| `/quit` | Exit |

## Other examples

| File | Description |
|------|-------------|
| `basic_usage.py` | Core MemoryFS API — no LLM needed |
| `openai_agent.py` | OpenAI function-calling loop (mock client included) |
| `langchain_agent.py` | LangChain integration |
| `agent_memory.py` | Multi-turn agent with memory |
| `code_search.py` | Code search use case |
| `infinite_context.py` | Long context compression |
| `multi_agent.py` | Multiple agents sharing memory |
