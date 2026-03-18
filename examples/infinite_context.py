"""
infinite_context.py — Demonstrates ContextMiddleware: automatic infinite context.

The Problem
-----------
Every LLM has a fixed context window (e.g., 128k tokens).  When it fills up,
traditional approaches either:
  (a) truncate old messages → information loss
  (b) summarise → lossy compression

The LLMFS Solution
------------------
ContextMiddleware acts like an OS virtual-memory manager:

  OS Concept          LLM Concept
  ──────────────────  ───────────────────────────────────────────
  RAM                 Context window
  Disk / swap         LLMFS (persistent storage)
  Page eviction       Offload low-importance turns to LLMFS
  Page fault          LLM calls memory_read / memory_search
  Page load           LLMFS returns exact, full-fidelity content
  MMU                 ContextManager (inside ContextMiddleware)

How it works (3 automatic steps each turn):
  1. Track tokens — count words in each new turn.
  2. Evict at 70% — when active tokens > 70% of max_tokens, offload
     the lowest-importance turns to LLMFS, targeting 50% usage after eviction.
  3. Inject index — prepend a compact "memory index" (~2k tokens) to the
     system prompt so the LLM knows what it can recall on demand.

Usage (two lines to add to any existing agent):
    from llmfs.context.middleware import ContextMiddleware
    agent = ContextMiddleware(my_agent, memory=MemoryFS())

Run:
    python examples/infinite_context.py
"""

from __future__ import annotations

import tempfile
import textwrap
from typing import Any

from llmfs import MemoryFS
from llmfs.context.middleware import ContextMiddleware


# ── Mock "LLM client" — replace with real openai / anthropic client ────────────

class MockChatModel:
    """Simulates a chat model that just echoes a summary of its inputs.

    In production, swap this for e.g.:
        class RealChatModel:
            def chat(self, messages):
                return openai.chat.completions.create(
                    model="gpt-4o", messages=messages
                ).choices[0].message
    """

    def __init__(self, name: str = "mock-gpt") -> None:
        self.name = name
        self._call_count = 0

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a canned assistant response that echoes context info."""
        self._call_count += 1
        # Count tokens (words) in everything passed to us
        total_tokens = sum(
            len(str(m.get("content", "")).split())
            for m in messages
        )
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else "(none)"

        # Check if memory index was injected (by ContextMiddleware)
        has_index = any(
            "LLMFS Memory Index" in str(m.get("content", ""))
            for m in messages
            if m.get("role") == "system"
        )

        response_text = (
            f"[Turn {self._call_count}] Responding to: '{last_user[:60]}'. "
            f"Received {len(messages)} messages ({total_tokens} tokens). "
            f"Memory index injected: {has_index}."
        )
        return {"role": "assistant", "content": response_text}


# ── Simulate a long conversation that fills the context window ─────────────────

# A realistic long-form conversation (turns are verbose to accumulate tokens)
CONVERSATION_TURNS = [
    "I'm starting a new microservices project. We'll use FastAPI for the HTTP layer, "
    "PostgreSQL 15 for the database, and Redis for session caching. The project is "
    "called Nexus and will handle B2B SaaS billing.",

    "Let's design the authentication module first. We need JWT tokens with 1-hour expiry, "
    "refresh tokens stored in Redis with 7-day TTL, and OAuth2 support for Google and GitHub. "
    "All tokens must be signed with RS256.",

    "For the database schema, users table should have: id UUID, email TEXT UNIQUE, "
    "password_hash TEXT, created_at TIMESTAMPTZ, plan_id FK, is_active BOOLEAN. "
    "Organisations table: id UUID, name TEXT, stripe_customer_id TEXT, created_at TIMESTAMPTZ.",

    "The billing module needs to integrate with Stripe. We should handle: subscription creation, "
    "plan upgrades/downgrades, invoice generation, webhook events (payment_succeeded, "
    "payment_failed, subscription_cancelled). Use idempotency keys for all Stripe calls.",

    "I found a bug in the refresh token logic! The Redis TTL is being reset on every "
    "read, not just on write. This means refresh tokens never expire. Fix: only set TTL "
    "on SETEX, not on GET. Also the JWT secret key is hardcoded — move to env var JWT_SECRET.",

    "Now let's set up the CI/CD pipeline. GitHub Actions: lint with ruff, type-check with "
    "mypy, test with pytest (min 80% coverage), build Docker image, push to GHCR, "
    "deploy to Fly.io on merge to main. Add staging environment gated by manual approval.",

    "The API rate limiting strategy: 100 req/min per IP for unauthenticated, "
    "1000 req/min per org for authenticated. Use sliding window algorithm in Redis. "
    "Return Retry-After header on 429. Log rate limit hits to audit_log table.",

    "Performance concern: the /invoices endpoint is doing N+1 queries. Need to add "
    "SELECT ... JOIN with line_items and organisations in a single query. Also add "
    "index on invoices(org_id, created_at DESC) for pagination.",

    "Security review findings: (1) Missing CORS policy — only allow our frontend domain. "
    "(2) SQL injection possible in search endpoint — use parameterised queries. "
    "(3) Passwords stored with bcrypt cost=12 — bump to 14 for new hashes, migrate lazily. "
    "(4) Add Content-Security-Policy and X-Frame-Options headers.",

    "Let's now write the deployment runbook. Pre-deploy: run db migrations, notify "
    "on-call team. Deploy: rolling update with health check on /health. Post-deploy: "
    "smoke test /health, /api/v1/ping, one Stripe webhook simulation. Rollback: "
    "fly releases rollback if error rate > 1% in 5 minutes.",

    "I want to add an audit log feature. Every write operation should record: who did it, "
    "what table/id was affected, old values (JSON), new values (JSON), timestamp, "
    "request IP and user agent. Store in audit_log table with partition by month.",

    "Final architecture decision: adopt hexagonal architecture (ports and adapters). "
    "Core domain knows nothing about FastAPI, PostgreSQL, Redis, or Stripe. "
    "All external systems are behind interfaces (ports). This makes unit testing trivial "
    "and allows swapping infrastructure without touching business logic.",
]


def run_demo(mem_path: str) -> None:
    mem = MemoryFS(path=mem_path)

    # Create a small mock model and wrap it with ContextMiddleware.
    # max_tokens=200 is intentionally tiny so we trigger eviction quickly
    # in this demo — in production you'd use 128_000 or your model's real limit.
    model = MockChatModel()
    agent = ContextMiddleware(
        model,
        memory=mem,
        max_tokens=200,       # deliberately tiny to trigger eviction in demo
        evict_at=0.70,        # evict when context hits 70% (140 tokens here)
        target_after_evict=0.50,  # evict down to 50% (100 tokens)
        session_id="nexus-project-session",
    )

    print("ContextMiddleware demo — tiny context window (200 tokens) to force evictions\n")
    print(f"Agent: {agent!r}\n")

    eviction_count = 0

    for i, user_text in enumerate(CONVERSATION_TURNS, start=1):
        stats_before = agent.get_context_stats()
        tokens_before = stats_before.get("total_tokens", 0)

        # Send the turn through the middleware
        response = agent.chat([{"role": "user", "content": user_text}])

        stats_after = agent.get_context_stats()
        tokens_after = stats_after.get("total_tokens", 0)

        # Detect eviction (token count dropped significantly)
        evicted_this_turn = tokens_after < tokens_before - 10
        if evicted_this_turn:
            eviction_count += 1
            eviction_marker = "  ← EVICTION TRIGGERED"
        else:
            eviction_marker = ""

        print(
            f"Turn {i:2d}: tokens {tokens_before:4d} → {tokens_after:4d}"
            f"  active_turns={stats_after.get('active_turns', '?')}"
            f"{eviction_marker}"
        )

    # ── Show what ended up in LLMFS ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Total evictions: {eviction_count}")
    print(f"Call count: {agent.get_context_stats()['call_count']}")

    active_turns = agent.get_active_turns()
    print(f"Turns still in active context: {len(active_turns)}")

    mem_status = mem.status()
    print(f"Turns offloaded to LLMFS: {mem_status['total']}")
    print(f"LLMFS disk usage: {mem_status['disk_mb']:.3f} MB")

    # ── Demonstrate that evicted content is still accessible ──────────────────
    print(f"\n{'─'*60}")
    print("Searching LLMFS for evicted content about 'JWT tokens':")
    results = mem.search("JWT token expiry refresh Redis", k=2)
    for r in results:
        snippet = r.chunk_text[:120].replace("\n", " ")
        print(f"  [{r.score:.3f}] {r.path}")
        print(f"           {snippet}…")

    # ── Show the current memory index injected into the system prompt ─────────
    index = agent.get_memory_index()
    if index:
        print(f"\n{'─'*60}")
        print("Memory index currently injected into system prompt:")
        print(textwrap.indent(index[:600], "  "))
        if len(index) > 600:
            print(f"  … ({len(index)} chars total)")
    else:
        print("\n(No memory index yet — eviction has not occurred or index is empty)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llmfs_infinite_") as tmp:
        run_demo(tmp)
