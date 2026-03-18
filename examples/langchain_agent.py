"""
langchain_agent.py — LLMFS with LangChain memory integrations.

Shows how to use:
  • LLMFSChatMemory      — drop-in for BaseChatMessageHistory
  • LLMFSRetrieverMemory — drop-in for BaseMemory with semantic retrieval

Since ``langchain`` may not be installed, this file uses fully working
mock fallbacks that reproduce the exact same interface.  When LangChain IS
installed, the real classes are used automatically.

Real-LangChain usage (uncomment when langchain is installed):
    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from llmfs.integrations.langchain import LLMFSChatMemory, LLMFSRetrieverMemory

    # --- LLMFSChatMemory ---
    chat_history = LLMFSChatMemory(memory_path="/tmp/llmfs")
    # plug into a ConversationChain or RunnableWithMessageHistory

    # --- LLMFSRetrieverMemory ---
    retriever_mem = LLMFSRetrieverMemory(memory_path="/tmp/llmfs", search_k=3)
    chain = ConversationChain(llm=ChatOpenAI(), memory=retriever_mem)
    chain.predict(input="What auth bugs did we find last week?")

Run:
    python examples/langchain_agent.py
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from typing import Any

from llmfs import MemoryFS


# ── Mock LangChain message types ───────────────────────────────────────────────
# These mirror the real langchain_core.messages interface exactly.
# When langchain is installed the real imports are used instead.

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # type: ignore
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content
        def __repr__(self) -> str:
            return f"HumanMessage(content={self.content!r})"

    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content
        def __repr__(self) -> str:
            return f"AIMessage(content={self.content!r})"

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content
        def __repr__(self) -> str:
            return f"SystemMessage(content={self.content!r})"


# ── Mock LLMFSChatMemory (same interface, no langchain dep) ───────────────────
# The real LLMFSChatMemory requires langchain; this mock demonstrates the same
# API using only LLMFS primitives.

class _MockLLMFSChatMemory:
    """Drop-in mock of LLMFSChatMemory for environments without langchain."""

    def __init__(self, mem: MemoryFS, session_prefix: str = "/session/chat") -> None:
        self._mem = mem
        self._prefix = session_prefix.rstrip("/")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    def add_user_message(self, content: str) -> None:
        self._mem.write(
            f"{self._prefix}/{self._ts()}",
            content,
            layer="session",
            tags=["human"],
            source="langchain",
        )

    def add_ai_message(self, content: str) -> None:
        self._mem.write(
            f"{self._prefix}/{self._ts()}",
            content,
            layer="session",
            tags=["ai"],
            source="langchain",
        )

    @property
    def messages(self) -> list[Any]:
        objs = self._mem.list(self._prefix, recursive=True)
        objs.sort(key=lambda o: o.metadata.created_at or "")
        result = []
        for obj in objs:
            role = "human"
            for tag in obj.tags:
                if tag in ("human", "user"):
                    role = "human"; break
                if tag in ("ai", "assistant"):
                    role = "ai"; break
            if role == "human":
                result.append(HumanMessage(content=obj.content))
            else:
                result.append(AIMessage(content=obj.content))
        return result

    def clear(self) -> None:
        self._mem.forget(layer="session")

    def __len__(self) -> int:
        return len(self._mem.list(self._prefix, recursive=True))


# ── Mock LLMFSRetrieverMemory ──────────────────────────────────────────────────

class _MockLLMFSRetrieverMemory:
    """Drop-in mock of LLMFSRetrieverMemory for environments without langchain."""

    def __init__(
        self,
        mem: MemoryFS,
        memory_key: str = "memory",
        search_k: int = 3,
        input_key: str = "input",
        session_prefix: str = "/session/turns",
    ) -> None:
        self._mem = mem
        self._memory_key = memory_key
        self._search_k = search_k
        self._input_key = input_key
        self._prefix = session_prefix.rstrip("/")

    @property
    def memory_variables(self) -> list[str]:
        """LangChain BaseMemory interface."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Search LLMFS and return injected context."""
        query = inputs.get(self._input_key, "")
        if not query.strip():
            return {self._memory_key: ""}

        results = self._mem.search(query, k=self._search_k)
        if not results:
            return {self._memory_key: ""}

        lines = ["## Relevant memories from LLMFS:"]
        for r in results:
            snippet = r.chunk_text[:300].replace("\n", " ")
            lines.append(f"- [{r.path}] (score={r.score:.2f}): {snippet}")

        return {self._memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Persist the input/output turn."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        human = inputs.get(self._input_key, "")
        ai = outputs.get("response") or outputs.get("output") or str(outputs)
        if human:
            self._mem.write(f"{self._prefix}/{ts}_human", human,
                            layer="session", tags=["human", "turn"])
        if ai:
            self._mem.write(f"{self._prefix}/{ts}_ai", ai,
                            layer="session", tags=["ai", "turn"])

    def clear(self) -> None:
        self._mem.forget(layer="session")


# ── Mock ConversationChain ─────────────────────────────────────────────────────

class MockConversationChain:
    """Mimics langchain.chains.ConversationChain with LLMFSRetrieverMemory.

    In production, replace with:
        from langchain.chains import ConversationChain
        from langchain.chat_models import ChatOpenAI
        chain = ConversationChain(llm=ChatOpenAI(), memory=retriever_mem)
    """

    def __init__(self, memory: _MockLLMFSRetrieverMemory) -> None:
        self._memory = memory
        self._responses = [
            "I remember we discussed JWT token expiry issues. The fix was changing "
            "the expiry from 0 to 3600 seconds in config.py.",
            "Based on our earlier work, the database has users, sessions, and audit_log tables "
            "with UUID primary keys. The auth bug has already been resolved.",
            "The retry decorator we wrote uses exponential backoff with 3 attempts by default. "
            "You can adjust max_attempts and backoff parameters.",
        ]
        self._idx = 0

    def predict(self, **kwargs: Any) -> str:
        """Simulate a chain call with memory retrieval."""
        user_input = kwargs.get("input", "")

        # 1. Load memory variables (semantic search)
        mem_vars = self._memory.load_memory_variables(kwargs)
        memory_ctx = mem_vars.get("memory", "")

        # 2. "Call LLM" — mock response
        response = self._responses[self._idx % len(self._responses)]
        self._idx += 1

        # 3. Save turn to memory
        self._memory.save_context(
            inputs={"input": user_input},
            outputs={"response": response},
        )

        # Show what memory was injected (for demo purposes)
        if memory_ctx:
            print(f"  [Memory injected ({len(memory_ctx)} chars)]")

        return response


# ── Demo: LLMFSChatMemory ──────────────────────────────────────────────────────

def demo_chat_memory(mem: MemoryFS) -> None:
    print("=" * 60)
    print("Demo 1: LLMFSChatMemory (persistent chat history)")
    print("=" * 60)

    # Use real integration if langchain is installed, mock otherwise
    if _LANGCHAIN_AVAILABLE:
        from llmfs.integrations.langchain import LLMFSChatMemory
        chat_mem = LLMFSChatMemory(mem=mem, session_prefix="/session/chat_demo")
    else:
        chat_mem = _MockLLMFSChatMemory(mem, session_prefix="/session/chat_demo")

    # Simulate a conversation
    exchanges = [
        ("Can you help me debug the auth module?",
         "Sure! Let me look at the JWT handler. I see the expiry is 0 — that's the bug."),
        ("What should the expiry be?",
         "For a web app, 3600 seconds (1 hour) is standard for access tokens."),
        ("Should we also add refresh tokens?",
         "Yes, store refresh tokens in Redis with 7-day TTL for better UX."),
    ]

    for human, ai in exchanges:
        chat_mem.add_user_message(human)
        chat_mem.add_ai_message(ai)

    messages = chat_mem.messages
    print(f"\nStored {len(messages)} messages:\n")
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role:5s}: {msg.content[:80]}")

    # Demonstrate persistence: create a new instance pointing to the same data
    if _LANGCHAIN_AVAILABLE:
        from llmfs.integrations.langchain import LLMFSChatMemory
        chat_mem2 = LLMFSChatMemory(mem=mem, session_prefix="/session/chat_demo")
    else:
        chat_mem2 = _MockLLMFSChatMemory(mem, session_prefix="/session/chat_demo")

    reloaded = chat_mem2.messages
    print(f"\n  (New instance, same path → reloaded {len(reloaded)} messages — persistence works ✓)")


# ── Demo: LLMFSRetrieverMemory ─────────────────────────────────────────────────

def demo_retriever_memory(mem: MemoryFS) -> None:
    print("\n" + "=" * 60)
    print("Demo 2: LLMFSRetrieverMemory (semantic retrieval)")
    print("=" * 60)

    # Pre-load some knowledge (simulating prior sessions)
    mem.write("/knowledge/auth/jwt", "JWT expiry fixed to 3600s. config.py line 42.",
              layer="knowledge", tags=["auth", "jwt"])
    mem.write("/knowledge/db/schema",
              "Users table: id UUID, email TEXT, password_hash TEXT. PostgreSQL 15.",
              layer="knowledge", tags=["database"])
    mem.write("/knowledge/utils/retry",
              "with_retry(max_attempts=3, backoff=1.0) — exponential backoff decorator.",
              layer="knowledge", tags=["utils"])

    # Create memory + chain
    if _LANGCHAIN_AVAILABLE:
        from llmfs.integrations.langchain import LLMFSRetrieverMemory
        retriever_mem = LLMFSRetrieverMemory(
            mem=mem,
            memory_key="memory",
            search_k=3,
            input_key="input",
            session_prefix="/session/retriever_demo",
        )
    else:
        retriever_mem = _MockLLMFSRetrieverMemory(
            mem,
            memory_key="memory",
            search_k=3,
            input_key="input",
            session_prefix="/session/retriever_demo",
        )

    chain = MockConversationChain(memory=retriever_mem)

    # Ask questions that require retrieving past knowledge
    questions = [
        "What was the JWT authentication bug and how was it fixed?",
        "Tell me about the database schema we're using.",
        "How does the retry logic work?",
    ]

    print()
    for q in questions:
        print(f"User : {q}")
        response = chain.predict(input=q)
        print(f"Chain: {response[:90]}…\n" if len(response) > 90 else f"Chain: {response}\n")

    # Show what's been saved
    turns = mem.list("/session/retriever_demo", recursive=True)
    print(f"  Saved {len(turns)} turn(s) to LLMFS session layer")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lc_status = "installed ✓" if _LANGCHAIN_AVAILABLE else "not installed — using mock fallback"
    print(f"LangChain: {lc_status}\n")

    with tempfile.TemporaryDirectory(prefix="llmfs_langchain_") as tmp:
        mem = MemoryFS(path=tmp)
        demo_chat_memory(mem)
        demo_retriever_memory(mem)

        status = mem.status()
        print(f"\nFinal: {status['total']} memories, {status['disk_mb']:.3f} MB")
