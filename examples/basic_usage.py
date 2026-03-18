"""
basic_usage.py — Quick-start demo for LLMFS.

Demonstrates the five core MemoryFS operations:
  write   → store content at a path
  search  → semantic similarity search
  read    → retrieve by exact path
  update  → append or replace content / change tags
  forget  → delete a specific memory

No OpenAI key or extra dependencies required — uses the built-in
local embedder (all-MiniLM-L6-v2, ~22 MB, runs on CPU).

Run:
    python examples/basic_usage.py
"""

import tempfile
from llmfs import MemoryFS

# ── 1. Create a MemoryFS instance backed by a temp directory ──────────────────
with tempfile.TemporaryDirectory(prefix="llmfs_demo_") as tmp:
    mem = MemoryFS(path=tmp)
    print(f"MemoryFS ready at: {tmp}\n")

    # ── 2. Write memories ─────────────────────────────────────────────────────
    mem.write(
        "/projects/auth/bug_report",
        content="JWT token expiry is set to 0 seconds — causes immediate logout on all endpoints.",
        layer="knowledge",
        tags=["bug", "auth", "jwt"],
    )
    mem.write(
        "/projects/auth/fix",
        content="Changed JWT_EXPIRY_SECONDS from 0 to 3600 in config.py line 42. Tested in staging.",
        layer="knowledge",
        tags=["fix", "auth", "jwt"],
    )
    mem.write(
        "/projects/db/schema",
        content="PostgreSQL 15. Main tables: users, sessions, audit_log. UUID primary keys.",
        layer="knowledge",
        tags=["database", "postgres"],
    )
    print("✓ Wrote 3 memories")

    # ── 3. Semantic search ────────────────────────────────────────────────────
    results = mem.search("authentication token problem", k=3)
    print(f"\nSearch 'authentication token problem' → {len(results)} result(s):")
    for r in results:
        print(f"  [{r.score:.3f}] {r.path}")
        print(f"           {r.chunk_text[:80]}...")

    # ── 4. Read by exact path ─────────────────────────────────────────────────
    obj = mem.read("/projects/auth/bug_report")
    print(f"\nRead /projects/auth/bug_report:")
    print(f"  content : {obj.content}")
    print(f"  tags    : {obj.tags}")
    print(f"  layer   : {obj.layer}")

    # ── 5. Update — append new information ───────────────────────────────────
    mem.update(
        "/projects/auth/bug_report",
        append="Reported by user @alice on 2025-01-10.",
        tags_add=["reported"],
    )
    updated = mem.read("/projects/auth/bug_report")
    print(f"\nAfter update, tags: {updated.tags}")
    print(f"  content: {updated.content[:120]}...")

    # ── 6. Forget a memory ────────────────────────────────────────────────────
    result = mem.forget("/projects/auth/fix")
    print(f"\nForgot /projects/auth/fix → {result}")

    # ── 7. Status summary ─────────────────────────────────────────────────────
    status = mem.status()
    print(f"\nStatus: {status['total']} memories, {status['disk_mb']:.2f} MB on disk")
    print("Done.")
