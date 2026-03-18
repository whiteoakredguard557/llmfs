"""
agent_memory.py — Simulated LLM agent with persistent cross-session memory.

Pattern:
  1. Before responding, the agent searches LLMFS for relevant past context.
  2. Each user message is stored as a memory so future sessions can find it.
  3. Each AI response is stored alongside the user message.
  4. On the next "session", the agent automatically finds prior context.

This mirrors exactly how a real LLM agent (GPT-4, Claude, etc.) would use
LLMFS — the only difference is you'd replace ``MockLLM`` with your actual
chat client and remove the canned responses.

Key insight: Because LLMFS persists to disk, the "Session 2" block finds
memories written during "Session 1" even though both run in the same process
here.  In production they'd be separate processes / separate days.

Run:
    python examples/agent_memory.py
"""

from __future__ import annotations

import tempfile
import textwrap
from datetime import datetime, timezone
from typing import Any

from llmfs import MemoryFS


# ── Mock LLM — swap for real client in production ─────────────────────────────

class MockLLM:
    """Simulates an LLM that returns canned responses.

    In a real integration, replace ``respond`` with:
        openai.chat.completions.create(model="gpt-4o", messages=messages)
    """

    def __init__(self, name: str = "MockGPT") -> None:
        self.name = name
        self._responses: list[str] = []
        self._idx = 0

    def queue(self, *responses: str) -> "MockLLM":
        """Pre-load canned responses (returned in order)."""
        self._responses.extend(responses)
        return self

    def respond(self, messages: list[dict[str, Any]]) -> str:
        """Return the next canned response (or a placeholder)."""
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return "[No canned response — would call real LLM here]"


# ── Memory-augmented agent loop ────────────────────────────────────────────────

class MemoryAgent:
    """An agent that persists every conversation turn to LLMFS.

    Before generating a response the agent calls ``mem.search()`` to
    retrieve relevant past context and prepends it to the messages list.
    This gives the agent effective memory across sessions without any
    context-window gymnastics.

    Args:
        llm:        Any object with a ``respond(messages) -> str`` method.
        memory:     MemoryFS instance (shared across sessions).
        agent_id:   Unique label used as a path namespace in LLMFS.
    """

    def __init__(self, llm: MockLLM, memory: MemoryFS, agent_id: str = "agent") -> None:
        self._llm = llm
        self._mem = memory
        self._agent_id = agent_id
        self._turn = 0

    def chat(self, user_message: str) -> str:
        self._turn += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        # ── Step 1: Persist the user message ──────────────────────────────────
        user_path = f"/{self._agent_id}/turns/{ts}_user"
        self._mem.write(
            user_path,
            content=user_message,
            layer="session",
            tags=["user", "turn"],
            source="agent",
        )

        # ── Step 2: Search for relevant past context ───────────────────────────
        past = self._mem.search(
            user_message,
            k=3,
            path_prefix=f"/{self._agent_id}",
        )

        # Build an augmented message list (inject retrieved memories)
        messages: list[dict[str, Any]] = []
        if past:
            memory_block = "\n".join(
                f"[{r.path}] {r.chunk_text[:200]}" for r in past
            )
            messages.append({
                "role": "system",
                "content": (
                    "## Relevant memories from previous turns:\n"
                    + memory_block
                ),
            })

        messages.append({"role": "user", "content": user_message})

        # ── Step 3: Call the LLM (mock here) ──────────────────────────────────
        response = self._llm.respond(messages)

        # ── Step 4: Persist the AI response ───────────────────────────────────
        ai_path = f"/{self._agent_id}/turns/{ts}_assistant"
        self._mem.write(
            ai_path,
            content=response,
            layer="session",
            tags=["assistant", "turn"],
            source="agent",
        )

        return response

    def recall(self, query: str, k: int = 5) -> list[Any]:
        """Explicit recall: search memory without generating a response."""
        return self._mem.search(query, k=k, path_prefix=f"/{self._agent_id}")


# ── Demo: two sessions separated by "time" ────────────────────────────────────

def run_demo(mem_path: str) -> None:
    mem = MemoryFS(path=mem_path)

    # ── Session 1 ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SESSION 1  (e.g. Monday morning)")
    print("=" * 60)

    llm1 = MockLLM("Session-1-LLM").queue(
        "I found the bug! The JWT expiry was set to 0 seconds in config.py line 42. "
        "Fixed it to 3600. Tests passing now.",

        "Good idea. I'll note that we should also rotate the secret key. "
        "Added to /knowledge/auth/todo for next sprint.",
    )
    agent1 = MemoryAgent(llm1, mem, agent_id="assistant")

    # First user message — no prior context yet
    msg1 = "Can you debug the auth module? Users are getting logged out immediately."
    print(f"\nUser : {msg1}")
    resp1 = agent1.chat(msg1)
    print(f"Agent: {textwrap.fill(resp1, 72, subsequent_indent='       ')}")

    msg2 = "Should we also update the secret key while we're in there?"
    print(f"\nUser : {msg2}")
    resp2 = agent1.chat(msg2)
    print(f"Agent: {textwrap.fill(resp2, 72, subsequent_indent='       ')}")

    # Promote key findings to the knowledge layer so they survive session cleanup
    mem.write(
        "/knowledge/auth/jwt_fix",
        content=(
            "JWT expiry was 0 seconds (config.py:42). Fixed to 3600. "
            "Secret key rotation flagged for next sprint."
        ),
        layer="knowledge",
        tags=["auth", "jwt", "fix"],
        source="agent",
    )
    print("\n✓ Session 1 done — persisted findings to /knowledge/auth/jwt_fix")

    # ── Session 2 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SESSION 2  (e.g. Wednesday — new process, same MemoryFS path)")
    print("=" * 60)

    # Same memory path → agent immediately has access to Session 1's memories
    llm2 = MockLLM("Session-2-LLM").queue(
        "Based on our previous work, the JWT fix in config.py is in place. "
        "The remaining item is secret key rotation — let me draft a migration plan.",
    )
    agent2 = MemoryAgent(llm2, mem, agent_id="assistant")

    msg3 = "What's the status of the auth work from earlier this week?"
    print(f"\nUser : {msg3}")
    resp3 = agent2.chat(msg3)
    print(f"Agent: {textwrap.fill(resp3, 72, subsequent_indent='       ')}")

    # Show what was recalled
    recalled = agent2.recall("auth JWT fix status", k=3)
    print(f"\n[Recalled {len(recalled)} memories for context]:")
    for r in recalled:
        print(f"  [{r.score:.3f}] {r.path}")

    # Final status
    status = mem.status()
    print(f"\nMemoryFS: {status['total']} total memories across {len(status['layers'])} layers")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llmfs_agent_") as tmp:
        run_demo(tmp)
