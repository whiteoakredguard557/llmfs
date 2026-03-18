# LLMFS — Implementation Plan

## What is LLMFS?

LLMFS is a filesystem-metaphor memory layer for LLMs and AI agents. It gives LLMs
persistent, searchable, structured memory — organized like a filesystem — so they can
store, retrieve, and relate knowledge across sessions, tasks, and agents.

Key insight: Instead of dumping everything into a context window (lossy, limited),
LLMs offload memory to LLMFS and retrieve only what they need, when they need it.
This enables "infinite context" — zero information loss, unlimited history.

---

## Core Concepts

### Memory as a Filesystem
- Memories are stored at paths: `/projects/auth/debug_session`, `/events/2026-03-15_fix`
- Directories = namespaces (projects, events, knowledge, session)
- Files = individual memory objects (`.mem` files internally)
- Human-readable paths make it intuitive for both LLMs and developers

### Memory Layers
| Layer        | Purpose                              | TTL              |
|--------------|--------------------------------------|------------------|
| `short_term` | Temporary reasoning scratch space    | Minutes (auto-expire) |
| `session`    | Current conversation context         | Session-scoped   |
| `knowledge`  | Persistent facts, learnings, code    | Permanent        |
| `events`     | Timestamped occurrences (bugs, fixes)| Permanent        |

### Memory Object Structure (Internal `.mem` file)
```json
{
  "id": "uuid-v4",
  "path": "/projects/powerscale/debug_session",
  "content": "User reports bucket creation failure...",
  "layer": "knowledge",
  "chunks": [
    {"index": 0, "text": "...", "embedding_id": "chroma-id-1"},
    {"index": 1, "text": "...", "embedding_id": "chroma-id-2"}
  ],
  "summaries": {
    "level_1": "Chunk-level summaries...",
    "level_2": "Document summary..."
  },
  "metadata": {
    "created_at": "2026-03-18T10:30:00Z",
    "modified_at": "2026-03-18T10:30:00Z",
    "accessed_at": "2026-03-18T10:35:00Z",
    "tags": ["debug", "powerscale", "s3"],
    "ttl": null,
    "source": "manual"
  },
  "relationships": [
    {"target": "/projects/powerscale/architecture", "type": "related_to", "strength": 0.85},
    {"target": "/events/2026-03-15_bug_fix", "type": "follows", "strength": 0.92}
  ]
}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Technology Stack

Component          Choice                   Rationale
Language           Python 3.11+             Rich ML ecosystem, fastest path to MVP
Primary interface  CLI + Python API         Broadest reach; FUSE as optional feature
Vector store       ChromaDB                 Embedded (no server), rich metadata filtering, SQLite backend
Metadata DB        SQLite (WAL mode)        Zero-config, concurrent reads, local-first
Default embedder   all-MiniLM-L6-v2         22MB, 1000+ queries/sec on CPU, no GPU needed
Chunking           Adaptive                 Code-aware (functions/classes) vs prose (headers/paragraphs)
Summarization      Extractive (TF-IDF)      Fast, deterministic, no GPU; optional LLM summarization via API
Packaging          pip install llmfs        Single command install, pyproject.toml
FUSE               Optional via fusepy      Linux/macOS only, pip install llmfs[fuse]
MCP Server         mcp Python SDK           Native LLM tool calling support

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Project Directory Structure

llmfs/
  llmfs/
    __init__.py                  # Public API: MemoryFS class
    core/
      filesystem.py              # MemoryFS main class — single entry point for all ops
      memory_object.py           # MemoryObject dataclass with all fields
      memory_layers.py           # Layer enum, TTL logic, layer-specific config
    embeddings/
      __init__.py
      base.py                    # Abstract EmbedderBase class
      local.py                   # SentenceTransformer (all-MiniLM-L6-v2) embedder
      openai.py                  # OpenAI text-embedding-3-small embedder
    storage/
      __init__.py
      vector_store.py            # ChromaDB wrapper — upsert, query, delete
      metadata_db.py             # SQLite wrapper — CRUD, search cache, relationships
    retrieval/
      __init__.py
      engine.py                  # Combines semantic + temporal + metadata filters
      ranker.py                  # Result re-ranking (score fusion, diversity)
    compression/
      __init__.py
      chunker.py                 # Adaptive chunker: code (AST) vs prose (headers)
      summarizer.py              # TF-IDF extractive summarizer, hierarchical levels
    graph/
      __init__.py
      memory_graph.py            # Relationship CRUD + BFS/DFS traversal
    query/
      __init__.py
      parser.py                  # MQL tokenizer + AST builder
      executor.py                # AST -> ChromaDB + SQLite queries
    context/
      __init__.py
      manager.py                 # ContextManager — the "Virtual Memory Manager"
      importance.py              # Importance scoring (0–1) for eviction priority
      extractor.py               # Artifact extraction: code blocks, errors, file refs
      index_builder.py           # Compact memory index generator (~2k tokens)
      middleware.py              # Drop-in middleware wrapping any LLM agent
    mcp/
      __init__.py
      server.py                  # MCP server (stdio + SSE transport)
      tools.py                   # 6 MCP tool definitions
      prompts.py                 # System prompt fragments telling LLMs how to use memory
    cli/
      __init__.py
      main.py                    # Click-based CLI entry point
      commands.py                # init, write, read, search, query, status, gc, serve, install-mcp
    integrations/
      __init__.py
      langchain.py               # LangChain BaseChatMessageHistory adapter
      openai_tools.py            # OpenAI function-calling JSON schemas
      fuse_mount.py              # Optional FUSE filesystem mount
  tests/
    __init__.py
    test_filesystem.py
    test_embeddings.py
    test_storage.py
    test_retrieval.py
    test_compression.py
    test_graph.py
    test_query.py
    test_context.py
    test_mcp.py
    test_cli.py
  examples/
    basic_usage.py               # Simple write/read/search
    agent_memory.py              # LLM agent with persistent memory
    code_search.py               # Ingest codebase, semantic search
    mcp_config.json              # Drop-in MCP config for Claude/Cursor
    langchain_agent.py           # LangChain agent using LLMFS memory
    openai_agent.py              # OpenAI function-calling with LLMFS
    multi_agent.py               # Two agents sharing memory
    infinite_context.py          # ContextMiddleware demo
  pyproject.toml
  README.md
  LICENSE
  .github/
    workflows/
      ci.yml                     # GitHub Actions: test, lint, build

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

SQLite Schema

-- Core file/memory registry
CREATE TABLE files (
    id            TEXT PRIMARY KEY,         -- UUID v4
    path          TEXT UNIQUE NOT NULL,     -- /projects/auth/debug
    name          TEXT NOT NULL,            -- debug
    layer         TEXT NOT NULL,            -- short_term | session | knowledge | events
    size          INTEGER,                  -- content byte size
    created_at    TEXT NOT NULL,            -- ISO 8601
    modified_at   TEXT NOT NULL,
    accessed_at   TEXT,
    content_hash  TEXT,                     -- SHA256 of content
    ttl_expires   TEXT,                     -- ISO 8601 or NULL
    source        TEXT DEFAULT 'manual'     -- manual | agent | mcp | cli
);

-- Embedding chunks for each file
CREATE TABLE chunks (
    id            TEXT PRIMARY KEY,
    file_id       TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    start_offset  INTEGER,
    end_offset    INTEGER,
    text          TEXT NOT NULL,
    embedding_id  TEXT NOT NULL,            -- ChromaDB document ID
    summary       TEXT                      -- Chunk-level extractive summary
);

-- Tag definitions
CREATE TABLE tags (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT UNIQUE NOT NULL
);

-- Many-to-many file <-> tags
CREATE TABLE file_tags (
    file_id  TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, tag_id)
);

-- Memory graph relationships
CREATE TABLE relationships (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,              -- related_to | follows | caused_by | contradicts
    strength    REAL DEFAULT 0.5,           -- 0.0 to 1.0
    created_at  TEXT NOT NULL
);

-- Search result cache (avoid re-embedding identical queries)
CREATE TABLE search_cache (
    query_hash    TEXT PRIMARY KEY,         -- SHA256 of (query + params)
    results_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_files_layer ON files(layer);
CREATE INDEX idx_files_path ON files(path);
CREATE INDEX idx_files_ttl ON files(ttl_expires);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

MQL — Memory Query Language

LLMFS ships a custom query language parsed into an AST, then translated to ChromaDB + SQLite.

-- Find memories by topic (semantic search in a path prefix)
SELECT memory FROM /projects WHERE topic = "authentication"

-- Time-scoped search
SELECT memory FROM /events WHERE date > 2026-01-01 AND date < 2026-03-01

-- Tag filter
SELECT memory FROM /knowledge WHERE tag = "s3" AND tag = "debug"

-- Graph traversal (BFS, depth 2)
SELECT memory FROM /projects RELATED TO "bug-fix-session" WITHIN 2

-- Combined semantic + metadata
SELECT memory FROM /knowledge WHERE SIMILAR TO "bucket creation error" AND tag = "s3" LIMIT 5

-- List recent memories
SELECT memory FROM /session ORDER BY created_at DESC LIMIT 10

MQL AST Node Types

  • SelectStatement: path, conditions, limit, order_by
  • SimilarCondition: query string → semantic search
  • TagCondition: exact tag match
  • DateCondition: temporal filter
  • TopicCondition: keyword/topic match → metadata filter
  • RelatedToCondition: graph traversal from anchor node
  • AndCondition / OrCondition: logical combinators

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

LLM Interface — The 6 Core Tools

These are exposed via MCP, OpenAI function calling, and the Python SDK.

memory_write

Store new knowledge or context.

memory_write(
    path: str,                    # e.g. "/projects/powerscale/debug"
    content: str,                 # the actual text to store
    layer: str = "knowledge",     # short_term | session | knowledge | events
    tags: list[str] = [],
    ttl_minutes: int | None = None  # auto-expire; use for short_term
) -> dict  # returns {id, path, status}

When LLM should use it: after learning something worth remembering (task results, user preferences, errors encountered, decisions made, code patterns).

memory_search

Semantic search across all memory.

memory_search(
    query: str,                    # natural language: "authentication bugs in JWT"
    layer: str | None = None,      # filter to specific layer
    tags: list[str] = [],          # filter by tags
    path_prefix: str | None = None,# restrict to subtree, e.g. "/projects"
    time_range: str | None = None, # "last 30 minutes", "today", "last 7 days"
    k: int = 5                     # number of results
) -> list[dict]  # [{path, content, score, metadata, tags}]

When LLM should use it: when it needs context it doesn't currently have, or before starting a new task.

memory_read

Read a specific memory by exact path.

memory_read(
    path: str,                     # exact path: "/projects/powerscale/debug"
    query: str | None = None       # optional: sub-query to retrieve only relevant chunks
) -> dict  # {path, content, metadata, tags, relationships}

When LLM should use it: when it knows the exact path (from the memory index).

memory_update

Modify an existing memory.

memory_update(
    path: str,
    content: str | None = None,    # full replace
    append: str | None = None,     # append to existing content
    tags_add: list[str] = [],
    tags_remove: list[str] = []
) -> dict  # {path, status, modified_at}

When LLM should use it: when facts change, corrections needed, or adding new findings to an existing memory.

memory_forget

Delete or expire memories.

memory_forget(
    path: str | None = None,       # specific memory path
    layer: str | None = None,      # wipe an entire layer
    older_than: str | None = None  # "7 days", "1 hour" — time-based cleanup
) -> dict  # {deleted_count, status}

When LLM should use it: outdated info, user requests deletion, TTL-based cleanup.

memory_relate

Link two memories in the knowledge graph.

memory_relate(
    source: str,                   # "/projects/debug_session"
    target: str,                   # "/knowledge/auth/architecture"
    relationship: str,             # "caused_by" | "related_to" | "follows" | "contradicts"
    strength: float = 0.8          # 0.0 to 1.0
) -> dict  # {relationship_id, status}

When LLM should use it: when two pieces of knowledge are connected, building a knowledge graph of a codebase or project.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Infinite Context — The Context Manager

This is the flagship feature. It turns LLMFS into a "virtual memory" system for LLMs, enabling unlimited effective context with zero information loss.

The Problem Today (Lossy Compression)

Turn 35: Context full (128k tokens)
         ↓
    Lossy Summarizer: 80k → 5k tokens
    94% of information LOST forever
         ↓
Turn 36: User: "What was the exact error on line 45?"
         LLM: "I don't have that detail anymore." ← FAILURE

The LLMFS Solution (Virtual Context)

OS Concept        →   LLM Concept
RAM               →   Context Window
Disk/Swap         →   LLMFS
Page eviction     →   Offload old turns to LLMFS (full fidelity)
Page fault        →   LLM calls memory_search/memory_read
Page load         →   LLMFS returns exact content
Virtual address   →   Memory path (/session/turns/12)
MMU               →   ContextManager

Context Window Layout (with LLMFS)

Context Window: 128k tokens
┌────────────┬─────────────────┬──────────────────────────────┐
│ System     │ Memory Index    │ Active Conversation          │
│ Prompt     │ (~2k tokens)    │ (recent 5–10 turns)          │
│ (~1k)      │ lists what's    │ (~20–80k tokens)             │
│            │ in LLMFS        │                              │
└────────────┴─────────────────┴──────────────────────────────┘
                    │
              ┌─────▼──────┐
              │  LLMFS     │  500k+ tokens stored, zero lost
              │            │  Full fidelity, semantically searchable
              └────────────┘

ContextManager Implementation Details

File: llmfs/context/manager.py

class ContextManager:
    def __init__(
        self,
        mem: MemoryFS,
        max_tokens: int = 128000,
        evict_at: float = 0.70,      # start evicting at 70% full
        target_after_evict: float = 0.50  # evict down to 50%
    ):
        self.mem = mem
        self.max_tokens = max_tokens
        self.evict_threshold = int(max_tokens * evict_at)
        self.target_tokens = int(max_tokens * target_after_evict)
        self.active_turns = []
        self.turn_counter = 0
        self.session_id = uuid4().hex

    def on_new_turn(self, role: str, content: str, tokens: int) -> None:
        """Called after every LLM message. Auto-evicts if needed."""
        ...

    def _evict(self) -> None:
        """Offload lowest-importance turns to LLMFS."""
        ...

    def _score_importance(self, content: str, role: str) -> float:
        """Score 0–1. Higher = keep in active context longer."""
        ...

    def _extract_artifacts(self, turn: dict) -> None:
        """Before eviction: extract and store code, errors, file refs separately."""
        ...

    def build_memory_index(self) -> str:
        """Generate compact index (~2k tokens) for injection into system prompt."""
        ...

    def get_active_turns(self) -> list[dict]:
        """Returns turns currently in the LLM context window."""
        ...

Importance Scoring Rules (llmfs/context/importance.py)

Signal                                                            Score Boost
Contains code block (```)                                         +0.20
Contains error/traceback                                          +0.20
Contains decision keyword (decided, plan, must, important)        +0.15
Role = user (user intent is critical)                             +0.10
Very recent turn (last 3)                                         +0.15
Very short content (<50 tokens)                                   -0.10
Conversational filler (ok, sure, thanks)                          -0.20

Artifact Extraction (llmfs/context/extractor.py)

Before a turn is evicted, structured artifacts are extracted and stored separately for precise retrieval:

  • Code blocks → /session/code/turn_{id}_{i} (tagged with language)
  • Stack traces / errors → /session/errors/turn_{id}
  • File paths mentioned → /session/files/turn_{id} (JSON list)
  • Decisions made → /session/decisions/turn_{id} (detected by keywords)
  • Full turn → /session/turns/{id} (always stored as fallback)

Memory Index Format (llmfs/context/index_builder.py)

The index is ~2k tokens, injected into the system prompt. Example output:

## LLMFS Memory Index
You have the following memories (use memory_read/memory_search to retrieve):

- [/session/turns/1]       (turn 1, 10:30) [user]      — User asked to fix auth module bug
- [/session/turns/2]       (turn 2, 10:31) [assistant] — Found JWT expiry at auth.py:45
- [/session/code/turn_2_0] (turn 2, 10:31) [code:py]   — Fixed auth.py token refresh logic
- [/session/errors/turn_3] (turn 3, 10:32) [error]     — TypeError: NoneType at auth.py:45
- [/session/turns/5]       (turn 5, 10:35) [user]      — Asked to also fix refresh endpoint
- [/session/code/turn_6_0] (turn 6, 10:36) [code:py]   — Updated refresh_token() method
... (17 more — use memory_search "topic" to find relevant ones)

Drop-In Middleware (llmfs/context/middleware.py)

Any existing agent gets infinite context with 2 lines:

from llmfs import MemoryFS
from llmfs.context import ContextMiddleware

agent = YourAgent(model="gpt-4o")
agent = ContextMiddleware(agent, memory=MemoryFS())  # done

# Middleware automatically:
# 1. Intercepts every turn (before + after)
# 2. Scores importance
# 3. Auto-evicts when context fills (at 70%)
# 4. Extracts artifacts before eviction
# 5. Rebuilds memory index after eviction
# 6. Injects index into system prompt
# 7. Provides memory_search/read tools to the LLM

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

MCP Server Design

File: llmfs/mcp/server.py

Exposes LLMFS as an MCP server (Model Context Protocol). Compatible with Claude, Cursor, Continue, Windsurf, and any MCP-compatible LLM client.

Transport: stdio (default) and SSE (for remote)

Start command:

llmfs serve --stdio        # stdio transport (for MCP clients)
llmfs serve --port 8765    # SSE transport (for remote clients)

Auto-install to Claude/Cursor:

llmfs install-mcp

Generates and writes:

{
  "mcpServers": {
    "llmfs": {
      "command": "llmfs",
      "args": ["serve", "--stdio"],
      "description": "AI memory filesystem — persistent, searchable, graph-linked memory"
    }
  }
}

System prompt fragment (llmfs/mcp/prompts.py): Shipped with LLMFS, tells LLMs exactly when/how to use each tool. Injected automatically when llmfs serve
is running.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

CLI Commands

# Initialize LLMFS in current directory (creates .llmfs/)
llmfs init [--path /custom/path]

# Write content to memory
llmfs write <path> <content>                        # direct content
llmfs write <path> --file ./report.md               # from file
llmfs write src/ /knowledge/myproject --recursive  # ingest directory
llmfs write <path> <content> --layer session --tags "auth,bug" --ttl 60

# Read a specific memory
llmfs read <path>
llmfs read <path> --query "what error occurred"     # focused read

# Semantic search
llmfs search "authentication logic"
llmfs search "bucket creation error" --layer knowledge --tags s3 --k 10
llmfs search "auth bug" --time "last 7 days"

# MQL structured query
llmfs query 'SELECT memory FROM /knowledge WHERE tag="auth" AND date > 2026-01-01'

# Update memory
llmfs update <path> --append "New finding: also affects refresh endpoint"
llmfs update <path> --content "Completely new content"
llmfs update <path> --tags-add "reviewed" --tags-remove "draft"

# Delete memory
llmfs forget <path>
llmfs forget --layer short_term
llmfs forget --older-than "7 days"

# Relate two memories
llmfs relate /events/debug_session /knowledge/auth/architecture related_to

# Show system status
llmfs status                    # total memories, layer breakdown, disk usage, index size

# Garbage collect (expired TTLs + orphaned chunks)
llmfs gc

# Start MCP server
llmfs serve [--stdio | --port 8765]

# Auto-configure MCP for Claude/Cursor
llmfs install-mcp [--client claude | cursor | continue]

# List memories
llmfs ls [path_prefix]          # list memories at a path
llmfs ls /knowledge --recursive

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Python API

from llmfs import MemoryFS

# Initialize (uses ~/.llmfs by default, or .llmfs in cwd)
mem = MemoryFS(path="~/.llmfs")

# Write
mem.write(
    path="/projects/auth/debug",
    content="User reports bucket creation failure...",
    layer="knowledge",
    tags=["debug", "s3"],
    ttl_minutes=None  # permanent
)

# Read
obj = mem.read("/projects/auth/debug")
obj.content       # full content
obj.metadata      # created_at, modified_at, tags, etc.
obj.relationships # linked memories

# Focused read (returns only relevant chunks)
obj = mem.read("/projects/auth/debug", query="what was the exact error")

# Search
results = mem.search("bucket creation error", k=5)
results = mem.search("auth bug", layer="events", tags=["jwt"], time_range="last 7 days")
# returns: List[SearchResult] with .path, .content, .score, .metadata

# Update
mem.update("/projects/auth/debug", append="Fixed in commit abc123")
mem.update("/projects/auth/debug", tags_add=["resolved"])

# Delete
mem.forget("/projects/auth/debug")
mem.forget(layer="short_term")
mem.forget(older_than="7 days")

# Relate
mem.relate(
    source="/events/debug_session",
    target="/knowledge/auth/architecture",
    relationship="related_to",
    strength=0.9
)

# List
memories = mem.list("/knowledge", recursive=True)

# MQL query
results = mem.query('SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug" LIMIT 5')

# Status
info = mem.status()  # {"total": 142, "layers": {...}, "disk_mb": 45.2}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

LangChain Integration

File: llmfs/integrations/langchain.py

from llmfs.integrations.langchain import LLMFSChatMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

memory = LLMFSChatMemory(memory_path="~/.llmfs")
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

# Memory automatically persists across sessions
# Past conversations are retrieved semantically

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

OpenAI Function Calling Integration

File: llmfs/integrations/openai_tools.py

from llmfs.integrations.openai_tools import LLMFS_TOOLS, llmfs_tool_handler

# LLMFS_TOOLS is a list of OpenAI-format tool definitions
# Pass to any OpenAI API call
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=LLMFS_TOOLS  # inject LLMFS tools
)

# Handle tool calls
result = llmfs_tool_handler(response.choices[0].message.tool_calls)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Comparison vs Competition

Feature                            mem0     Letta  ChromaDB  LLMFS
Filesystem metaphor                No       No     No        Yes
Memory layers                      Partial  Yes    No        Yes
Memory graph                       No       No     No        Yes
Query language (MQL)               No       No     SQL-like  Custom MQL
Auto-compression                   No       Yes    No        Yes
Infinite context (VM model)        No       No     No        Yes
CLI interface                      No       Yes    No        Yes
Local-first (no server)            No       No     Yes       Yes
Zero-config                        No       No     Partial   Yes (llmfs init)
MCP server built-in                No       No     No        Yes
FUSE mount                         No       No     No        Yes (optional)
Drop-in agent middleware           No       No     No        Yes

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Implementation Phases

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 1 — MVP Core (Days 1–3)

Goal: A working pip install llmfs with basic write/read/search from CLI and Python.

1.1 Project Scaffolding

  • [ ] Create pyproject.toml with all dependencies declared
    • chromadb, sentence-transformers, click, sqlite3 (stdlib), numpy, scikit-learn
    • Optional extras: [fuse] → fusepy; [openai] → openai; [langchain] → langchain
  • [ ] Create package structure (all __init__.py files)
  • [ ] Set up pytest + pytest-cov in dev dependencies
  • [ ] Create .gitignore, LICENSE (MIT), README.md skeleton

1.2 MemoryObject Dataclass (core/memory_object.py)

  • [ ] Fields: id, path, content, layer, chunks, summaries, metadata, relationships
  • [ ] metadata sub-object: created_at, modified_at, accessed_at, tags, ttl, source
  • [ ] to_dict() / from_dict() serialization
  • [ ] content_hash property (SHA256)
  • [ ] Validation: path must start with /, layer must be valid enum

1.3 Memory Layers (core/memory_layers.py)

  • [ ] MemoryLayer enum: SHORT_TERM, SESSION, KNOWLEDGE, EVENTS
  • [ ] Default TTL per layer: SHORT_TERM=60min, SESSION=None (session-scoped), KNOWLEDGE=None, EVENTS=None
  • [ ] is_expired(memory_obj) function
  • [ ] Layer config: default tags, eviction priority order

1.4 SQLite Metadata DB (storage/metadata_db.py)

  • [ ] MetadataDB class wrapping sqlite3
  • [ ] initialize(): creates tables + indexes (schema as above)
  • [ ] WAL mode enabled: PRAGMA journal_mode=WAL
  • [ ] CRUD: insert_file(), get_file(), update_file(), delete_file()
  • [ ] list_files(layer, path_prefix, tags, created_after, created_before)
  • [ ] insert_chunk(), get_chunks(file_id)
  • [ ] insert_tag(), get_or_create_tag(), tag_file(), untag_file()
  • [ ] insert_relationship(), get_relationships(file_id)
  • [ ] expire_ttl(): delete all files where ttl_expires < now()
  • [ ] search_cache_get(query_hash), search_cache_set(query_hash, results, ttl_seconds=300)

1.5 ChromaDB Vector Store (storage/vector_store.py)

  • [ ] VectorStore class wrapping ChromaDB
  • [ ] One ChromaDB collection per LLMFS instance (named llmfs_<hash_of_path>)
  • [ ] upsert(embedding_id, embedding, metadata, text)
  • [ ] query(embedding, k, where_filter) — returns [(id, text, score, metadata)]
  • [ ] delete(embedding_id), delete_by_file_id(file_id)
  • [ ] Metadata stored per chunk: file_id, path, layer, chunk_index, tags (comma-joined)
  • [ ] where_filter maps to ChromaDB metadata filters

1.6 Local Embedder (embeddings/local.py)

  • [ ] LocalEmbedder(EmbedderBase) using sentence-transformers
  • [ ] Model: all-MiniLM-L6-v2 (auto-download on first use)
  • [ ] embed(text: str) -> list[float]
  • [ ] embed_batch(texts: list[str]) -> list[list[float]]
  • [ ] Cache embeddings for identical strings (in-memory LRU, size=1000)
  • [ ] EmbedderBase abstract class: embed(), embed_batch(), model_name property

1.7 Adaptive Chunker (compression/chunker.py)

  • [ ] AdaptiveChunker class
  • [ ] chunk(content: str, content_type: str) -> list[Chunk]
  • [ ] Auto-detect content type: Python code (detect def , class ), Markdown (detect #), plain text
  • [ ] Code chunking: split on top-level function/class definitions (using ast module for Python; regex fallback for others)
    • Chunk size target: 512 tokens (use len(text.split()) as proxy)
    • Overlap: 1 function signature as context header
  • [ ] Prose chunking: split by headers (#, ##), then by paragraphs if too large
    • Chunk size target: 256 tokens
    • Overlap: last sentence of previous chunk
  • [ ] Plain text: sliding window, 256 tokens, 50-token overlap
  • [ ] Chunk dataclass: index, text, start_offset, end_offset

1.8 MemoryFS Core (core/filesystem.py)

  • [ ] MemoryFS class — the single public API
  • [ ] Constructor: MemoryFS(path="~/.llmfs", embedder=None)
    • Auto-creates .llmfs/ directory structure
    • Initializes MetadataDB, VectorStore, LocalEmbedder, AdaptiveChunker
    • Runs expire_ttl() on startup
  • [ ] write(path, content, layer, tags, ttl_minutes, source) -> MemoryObject
    • Validate path format
    • Hash content, check if unchanged (skip re-embed if same)
    • Chunk content via AdaptiveChunker
    • Embed all chunks via embedder
    • Upsert chunks into VectorStore
    • Insert/update metadata in MetadataDB
    • Return MemoryObject
  • [ ] read(path, query=None) -> MemoryObject
    • Fetch from MetadataDB
    • If query provided: embed query, retrieve only top-k matching chunks
    • Update accessed_at
    • Return MemoryObject
  • [ ] search(query, layer, tags, path_prefix, time_range, k) -> list[SearchResult]
    • Check search_cache first
    • Embed query
    • Build ChromaDB where_filter from layer/tags/path_prefix
    • Apply time_range filter via metadata (created_at range)
    • Query VectorStore, get chunk results
    • Deduplicate to file level (take best chunk score per file)
    • Store in cache
    • Return list[SearchResult]
  • [ ] update(path, content, append, tags_add, tags_remove) -> MemoryObject
    • If append: load existing, concatenate, re-chunk, re-embed
    • If content: full replace, re-chunk, re-embed
    • Handle tag mutations via MetadataDB
  • [ ] forget(path, layer, older_than) -> dict
    • Delete from VectorStore (by file_id or filter)
    • Delete from MetadataDB
    • Return {deleted_count, status}
  • [ ] relate(source, target, relationship, strength) -> dict
    • Validate both paths exist
    • Insert into relationships table
  • [ ] list(path_prefix, recursive, layer) -> list[MemoryObject]
  • [ ] status() -> dict
    • Total memories per layer
    • Total chunks
    • Disk usage of .llmfs/ directory
    • ChromaDB collection size
    • Cache hit rate

1.9 Basic CLI (cli/)

  • [ ] main.py: Click group @click.group(), entry point llmfs
  • [ ] llmfs init: create .llmfs/ in cwd, print success + next steps
  • [ ] llmfs write <path> <content>: basic write with --layer, --tags, --ttl, --file
  • [ ] llmfs read <path>: print memory content + metadata
  • [ ] llmfs search <query>: print ranked results with scores
  • [ ] llmfs forget <path>: delete with confirmation prompt
  • [ ] llmfs status: print summary table
  • [ ] Output formatting: use rich for tables and colored output

1.10 Tests for Phase 1

  • [ ] test_memory_object.py: dataclass, serialization, validation
  • [ ] test_storage.py: SQLite CRUD, ChromaDB upsert/query/delete
  • [ ] test_embeddings.py: local embedder, batch, caching
  • [ ] test_chunker.py: code chunking, prose chunking, plain text
  • [ ] test_filesystem.py: write/read/search/update/forget/relate
  • [ ] test_cli.py: CLI commands via Click's CliRunner

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 2 — Memory Intelligence (Days 3–5)

Goal: Memory layers with TTL, extractive summarization, memory graph traversal, time-based retrieval.

2.1 Memory Layer Enforcement

  • [ ] On write(): enforce layer-specific defaults (auto-set TTL for short_term)
  • [ ] Background TTL expiry: MemoryFS._gc_expired() called on every write() (throttled to once/minute)
  • [ ] session layer cleanup: mem.forget(layer="session") at end of session
  • [ ] Layer isolation in search: search(layer="knowledge") only returns from that layer

2.2 Extractive Summarizer (compression/summarizer.py)

  • [ ] ExtractiveSummarizer class
  • [ ] summarize_chunk(text: str, max_sentences=3) -> str
    • TF-IDF sentence scoring using scikit-learn
    • Return top N sentences by score (preserving original order)
  • [ ] summarize_document(chunks: list[str], max_sentences=5) -> str
    • Summarize across all chunks
    • Used as level_2 summary in MemoryObject.summaries
  • [ ] Store summaries in MemoryObject.summaries on write
  • [ ] summarize_chunks_individually(chunks) -> list[str]: level_1 summaries
  • [ ] Summary used in memory index display (80-char truncation)

2.3 Memory Graph (graph/memory_graph.py)

  • [ ] MemoryGraph class backed by SQLite relationships table
  • [ ] add_edge(source_path, target_path, rel_type, strength)
  • [ ] get_neighbors(path, rel_type=None) -> list[dict]
  • [ ] traverse(start_path, max_depth=2, rel_types=None) -> list[str]
    • BFS traversal returning all paths within depth
  • [ ] shortest_path(source_path, target_path) -> list[str]
  • [ ] find_related(path, k=5) -> list[dict] — returns by strength score
  • [ ] Auto-relate on write: if semantic similarity > 0.85 with existing memory, auto-create related_to edge (optional, configurable)

2.4 Time-Based Retrieval

  • [ ] search(time_range="last 30 minutes") — parse human time strings
  • [ ] Time string parser: "last N minutes", "last N hours", "last N days", "today", "yesterday", "this week"
  • [ ] Translates to created_at >= <computed timestamp> SQLite filter
  • [ ] list(path_prefix, since="today") — time-filtered listing

2.5 Retrieval Engine (retrieval/engine.py)

  • [ ] RetrievalEngine combining all retrieval modes
  • [ ] semantic_search(query, k, filters) -> list[SearchResult]
  • [ ] temporal_search(time_range, layer, path_prefix) -> list[SearchResult]
  • [ ] graph_search(start_path, depth, query) -> list[SearchResult]
    • Traverse graph from start, then semantic re-rank results
  • [ ] hybrid_search(query, time_range, tags, layer, graph_start, k) -> list[SearchResult]
    • Combine semantic + temporal + graph results
    • Deduplicate and merge scores

2.6 Result Ranker (retrieval/ranker.py)

  • [ ] Ranker class
  • [ ] rank(results: list[SearchResult], query: str) -> list[SearchResult]
  • [ ] Scoring factors:
    • Semantic similarity score (from ChromaDB)
    • Recency boost: exp(-age_hours / decay_rate), default decay_rate=168 (1 week)
    • Access frequency boost: log(1 + access_count)
    • Tag match boost: +0.1 per exact tag match
    • Layer priority: knowledge > events > session > short_term
  • [ ] Reciprocal Rank Fusion (RRF) for combining multiple ranked lists
  • [ ] diversity_rerank(results, lambda=0.5) — MMR-style diversity

2.7 CLI Additions (Phase 2)

  • [ ] llmfs query '<MQL statement>'
  • [ ] llmfs ls [path] — list memories
  • [ ] llmfs gc — garbage collect expired memories
  • [ ] llmfs relate <source> <target> <relationship>
  • [ ] llmfs update <path> --append "..."

2.8 Tests for Phase 2

  • [ ] test_summarizer.py: chunk summary, document summary
  • [ ] test_graph.py: add edges, BFS traversal, shortest path
  • [ ] test_retrieval.py: semantic, temporal, hybrid, ranker
  • [ ] test_layers.py: TTL expiry, layer isolation, session cleanup

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 3 — Infinite Context + MCP Server (Days 5–7)

Goal: ContextManager for infinite context, full MCP server for LLM-native access.

3.1 Importance Scorer (context/importance.py)

  • [ ] ImportanceScorer class
  • [ ] score(content: str, role: str, turn_index: int, total_turns: int) -> float
  • [ ] Rules (additive, clamped to [0, 1]):
    • Base: 0.5
    • Code block (```): +0.20
    • Error/traceback: +0.20
    • Decision keywords: +0.15
    • User role: +0.10
    • Very recent (last 3 turns): +0.15
    • Short filler content: -0.20
  • [ ] Configurable weights via constructor

3.2 Artifact Extractor (context/extractor.py)

  • [ ] ArtifactExtractor class
  • [ ] extract(turn: dict, mem: MemoryFS, session_id: str) -> list[str]
    • Returns list of paths written to LLMFS
  • [ ] Extract code blocks: regex ```lang\n...\n```
    • Write to /session/{session_id}/code/turn_{id}_{i}
    • Tag with ["code", lang]
  • [ ] Extract errors/stack traces: detect Traceback, Error:, Exception:
    • Write to /session/{session_id}/errors/turn_{id}
    • Tag with ["error"]
  • [ ] Extract file paths: regex [\w/.-]+\.(py|js|ts|go|rs|java|cpp|h|json|yaml|toml)
    • Write to /session/{session_id}/files/turn_{id} (JSON list)
    • Tag with ["file_references"]
  • [ ] Extract decisions: detect "decided", "will use", "going with", "approach:", "plan:"
    • Write to /session/{session_id}/decisions/turn_{id}
    • Tag with ["decision"]

3.3 Memory Index Builder (context/index_builder.py)

  • [ ] IndexBuilder class
  • [ ] build(session_id: str, mem: MemoryFS, max_entries=50) -> str
    • List all memories under /session/{session_id}/
    • Format each as one line: [path] (turn N, HH:MM) [role] — 80-char summary
    • Sort by turn_id ascending
    • If more than max_entries: show first 10 + last 10, note (N more in between)
    • Total output: ~2k tokens
  • [ ] estimate_tokens(text: str) -> int — rough len(text) // 4

3.4 Context Manager (context/manager.py)

  • [ ] ContextManager class (full implementation as designed above)
  • [ ] on_new_turn(role, content, tokens): track turn, check threshold, evict if needed
  • [ ] _evict(): sort by importance, evict lowest until at target, call _extract_artifacts, write full turn
  • [ ] _rebuild_index(): call IndexBuilder.build(), store as self._memory_index
  • [ ] get_system_prompt_addon() -> str: return current memory index for injection
  • [ ] get_active_turns() -> list[dict]: turns currently in context
  • [ ] reset_session(): clear session layer, reset active turns
  • [ ] Configurable: max_tokens, evict_at, target_after_evict, session_id

3.5 Context Middleware (context/middleware.py)

  • [ ] ContextMiddleware class
  • [ ] Wraps any object with a chat(messages) -> response or invoke(input) -> output interface
  • [ ] __init__(agent, memory: MemoryFS, max_tokens=128000)
  • [ ] Intercepts calls: inject memory index into system message, track turns, call on_new_turn
  • [ ] Works with: OpenAI client, LangChain chains, raw function calls
  • [ ] get_context_stats() -> dict: token usage, evicted turns, cache hits

3.6 MCP Server (mcp/server.py and mcp/tools.py)

  • [ ] Use mcp Python package (pip install mcp)
  • [ ] LLMFSMCPServer class
  • [ ] Register all 6 tools: memory_write, memory_search, memory_read, memory_update, memory_forget, memory_relate
  • [ ] Each tool has: name, description, inputSchema (JSON Schema), async handler
  • [ ] Tool handlers call MemoryFS methods
  • [ ] Error handling: return {error: "...", status: "error"} on failure
  • [ ] run_stdio(): start server on stdin/stdout (for MCP clients)
  • [ ] run_sse(port): start HTTP+SSE server (for remote)
  • [ ] CLI: llmfs serve [--stdio | --port N]

3.7 System Prompt (mcp/prompts.py)

  • [ ] LLMFS_SYSTEM_PROMPT constant
  • [ ] Covers: when to use each tool, layer descriptions, path naming conventions, example calls
  • [ ] get_prompt(include_index=False, mem=None) -> str: optionally append memory index

3.8 CLI: install-mcp

  • [ ] Detect OS and common config locations:
    • Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
    • Cursor: ~/.cursor/mcp.json
    • Continue: ~/.continue/config.json
  • [ ] llmfs install-mcp --client claude: write/merge MCP server config
  • [ ] llmfs install-mcp --print: just print the JSON to stdout

3.9 Tests for Phase 3

  • [ ] test_importance.py: scoring rules, edge cases
  • [ ] test_extractor.py: code block extraction, error extraction, file refs
  • [ ] test_index_builder.py: index format, token estimation, truncation
  • [ ] test_context_manager.py: eviction trigger, artifact extraction, index rebuild
  • [ ] test_middleware.py: wrapping an agent, turn tracking, eviction
  • [ ] test_mcp.py: tool registration, handler calls, error cases

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 4 — MQL + Agent Integrations + FUSE (Days 7–9)

Goal: Full MQL query language, LangChain/OpenAI integrations, optional FUSE mount.

4.1 MQL Parser (query/parser.py)

  • [ ] Tokenizer: splits MQL string into tokens (SELECT, FROM, WHERE, AND, OR, SIMILAR TO, RELATED TO, WITHIN, LIMIT, ORDER BY, comparison operators, 
  string literals, identifiers)
  • [ ] Parser: tokens → AST
  • [ ] AST node classes:
    • SelectStatement(path, conditions, limit, order_by)
    • SimilarCondition(query_str)
    • TagCondition(tag_name, op)
    • DateCondition(field, op, value)
    • TopicCondition(topic_str)
    • RelatedToCondition(anchor_path, depth)
    • AndCondition(left, right) / OrCondition(left, right)
  • [ ] Error handling: MQLParseError with line/column info

4.2 MQL Executor (query/executor.py)

  • [ ] MQLExecutor class
  • [ ] execute(ast: SelectStatement, mem: MemoryFS) -> list[SearchResult]
  • [ ] AST → retrieval calls:
    • SimilarCondition → mem.search(query)
    • TagCondition → SQLite tag filter
    • DateCondition → SQLite date range
    • RelatedToCondition → MemoryGraph.traverse()
    • Combined conditions → RetrievalEngine.hybrid_search()
  • [ ] mem.query(mql_string) → parse + execute

4.3 LangChain Integration (integrations/langchain.py)

  • [ ] LLMFSChatMemory(BaseChatMessageHistory):
    • add_message(message): write to /session/chat/{timestamp}
    • messages property: search recent session turns
    • clear(): forget session layer
  • [ ] LLMFSRetrieverMemory(BaseMemory):
    • load_memory_variables(inputs): semantic search on the query, inject as context
    • save_context(inputs, outputs): write turn to LLMFS
  • [ ] Both work as drop-in replacements for LangChain's built-in memory classes

4.4 OpenAI Tools Integration (integrations/openai_tools.py)

  • [ ] LLMFS_TOOLS: list of OpenAI-format tool definitions (JSON Schema for all 6 tools)
  • [ ] LLMFSToolHandler(mem: MemoryFS):
    • handle(tool_call) -> str: dispatches to correct MemoryFS method
    • handle_batch(tool_calls) -> list[str]
  • [ ] Example showing full function-calling loop with LLMFS

4.5 FUSE Mount (integrations/fuse_mount.py)

  • [ ] Optional, only if fusepy installed
  • [ ] LLMFSFuse(LoggingMixIn, Operations):
    • readdir(path, fh): list memories as files
    • read(path, size, offset, fh): return memory content
    • write(path, data, offset, fh): write content to LLMFS
    • create(path, mode): create new empty memory
    • unlink(path): delete memory
    • getattr(path, fh=None): stat-like info
  • [ ] llmfs mount <mountpoint>: CLI command to start FUSE
  • [ ] llmfs unmount <mountpoint>: unmount
  • [ ] Install instructions: pip install llmfs[fuse]

4.6 Tests for Phase 4

  • [ ] test_mql_parser.py: tokenizer, parser, all AST node types, error cases
  • [ ] test_mql_executor.py: all condition types, combined conditions
  • [ ] test_langchain.py: chat memory, retriever memory
  • [ ] test_openai_tools.py: tool definitions, handler dispatch

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 5 — Polish for GitHub (Day 9–10)

Goal: Production-ready open source release.

5.1 Complete Test Suite

  • [ ] All phases: target 90%+ code coverage
  • [ ] Integration tests: full end-to-end write → search → retrieve cycle
  • [ ] Performance tests: search latency < 100ms for 10k memories
  • [ ] pytest.ini configuration: markers, coverage settings
  • [ ] GitHub Actions ci.yml: run tests on push/PR, Python 3.11 + 3.12

5.2 README.md

  • [ ] Elevator pitch (3 sentences)
  • [ ] Quick install + init + first write/search (copy-pasteable)
  • [ ] Architecture diagram (ASCII or image)
  • [ ] Feature table (vs competition)
  • [ ] Usage sections: CLI, Python API, MCP setup, LangChain, OpenAI, ContextMiddleware
  • [ ] Examples: basic, agent memory, code search, infinite context
  • [ ] Configuration reference
  • [ ] Contributing guide

5.3 Examples Directory

  • [ ] basic_usage.py: write/read/search in 20 lines
  • [ ] agent_memory.py: LLM agent that stores and retrieves across sessions
  • [ ] code_search.py: ingest a codebase, semantic search
  • [ ] infinite_context.py: ContextMiddleware wrapping OpenAI client
  • [ ] multi_agent.py: two agents sharing LLMFS memory
  • [ ] langchain_agent.py: ConversationChain with LLMFS memory
  • [ ] openai_agent.py: function-calling agent with LLMFS tools
  • [ ] mcp_config.json: ready-to-use MCP config for Claude/Cursor

5.4 pyproject.toml (Final)

[project]
name = "llmfs"
version = "0.1.0"
description = "A filesystem-metaphor memory layer for LLMs and AI agents"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "mcp>=0.1.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0.0"]
langchain = ["langchain>=0.1.0", "langchain-community>=0.0.1"]
fuse = ["fusepy>=3.0.1"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "pytest-asyncio>=0.21"]

[project.scripts]
llmfs = "llmfs.cli.main:cli"

5.5 GitHub Actions (ci.yml)

  • [ ] Trigger: push and pull_request to main
  • [ ] Matrix: Python 3.11, 3.12
  • [ ] Steps: checkout, setup Python, pip install -e .[dev], pytest --cov, upload coverage
  • [ ] Lint: ruff check .
  • [ ] Type check: mypy llmfs/ (optional, progressive)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Key Implementation Notes

Directory Layout on Disk

~/.llmfs/                  # default storage location
  metadata.db              # SQLite database
  chroma/                  # ChromaDB persistence directory
  config.json              # LLMFS configuration

Or project-local: .llmfs/ in current directory.

Configuration (config.json)

{
  "embedder": "local",               // local | openai
  "embedder_model": "all-MiniLM-L6-v2",
  "chunk_size_tokens": 256,
  "chunk_overlap_tokens": 50,
  "search_cache_ttl_seconds": 300,
  "auto_relate_threshold": 0.85,
  "context_manager": {
    "max_tokens": 128000,
    "evict_at": 0.70,
    "target_after_evict": 0.50
  },
  "layers": {
    "short_term": {"ttl_minutes": 60},
    "session": {"ttl_minutes": null},
    "knowledge": {"ttl_minutes": null},
    "events": {"ttl_minutes": null}
  }
}

Error Handling Strategy

  • MemoryNotFoundError: path doesn't exist
  • MemoryWriteError: failed to write (storage issue)
  • MQLParseError: invalid MQL syntax
  • EmbedderError: embedding model failed
  • All errors: logged to ~/.llmfs/llmfs.log, raised to caller

Thread Safety

  • SQLite WAL mode: multiple readers, single writer
  • ChromaDB: embedded, handles its own locking
  • MemoryFS: not thread-safe by default; use MemoryFS(thread_safe=True) for threading.Lock on writes

Performance Targets

  • Write (single memory, ~500 tokens): < 200ms
  • Search (10k memories, top-5): < 100ms
  • Read (by path): < 10ms
  • Context eviction (20 turns): < 500ms
  • MQL query: < 200ms

---