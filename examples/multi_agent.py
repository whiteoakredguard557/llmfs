"""
multi_agent.py — Two agents sharing a single MemoryFS instance.

Demonstrates the "shared memory bus" pattern:

  Agent A  ──write──▶  MemoryFS  ◀──search──  Agent B
                           │
                       (on disk)
                     both agents read
                     each other's writes

Use cases:
  • Researcher agent + Writer agent: researcher stores findings, writer reads them.
  • Planner agent + Executor agent: planner writes a plan, executor reads and acts.
  • Multiple parallel agents working on different sub-tasks, sharing results.

No real API keys needed — uses mock agents with canned responses.

Run:
    python examples/multi_agent.py
"""

from __future__ import annotations

import tempfile
import textwrap
from datetime import datetime, timezone
from typing import Any

from llmfs import MemoryFS


# ── Mock base agent ────────────────────────────────────────────────────────────

class BaseAgent:
    """A minimal agent that can read from and write to a shared MemoryFS.

    In production, replace ``_generate`` with a real LLM call.
    """

    def __init__(
        self,
        name: str,
        memory: MemoryFS,
        namespace: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._mem = memory
        self._ns = namespace.rstrip("/")  # e.g. "/agents/researcher"
        self._call_count = 0

    # ── Memory helpers ─────────────────────────────────────────────────────────

    def remember(
        self,
        key: str,
        content: str,
        tags: list[str] | None = None,
        layer: str = "knowledge",
    ) -> str:
        """Write a memory under this agent's namespace.  Returns the full path."""
        path = f"{self._ns}/{key}"
        self._mem.write(path, content, layer=layer, tags=tags or [], source=self.name)
        return path

    def recall(self, query: str, k: int = 5, path_prefix: str | None = None) -> list[Any]:
        """Semantic search — optionally scoped to a namespace."""
        return self._mem.search(query, k=k, path_prefix=path_prefix or "/")

    def read_shared(self, path: str) -> str:
        """Read a specific memory by exact path (possibly written by another agent)."""
        try:
            obj = self._mem.read(path)
            return obj.content
        except Exception as exc:
            return f"[error reading {path}: {exc}]"

    # ── Stub for a real LLM call ───────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Stub: replace with openai.chat.completions.create(…) in production."""
        self._call_count += 1
        return f"[{self.name}] (mock response to: {prompt[:60]}…)"


# ── Researcher agent ───────────────────────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """Investigates a topic and stores structured findings in LLMFS."""

    def investigate(self, topic: str) -> dict[str, str]:
        """Run a (mock) investigation and persist findings."""
        print(f"\n[{self.name}] Investigating: {topic!r}")

        # In production: query external APIs, run code, call LLM, etc.
        findings = {
            "summary": (
                f"Research on '{topic}': Found 3 key patterns. "
                "Performance bottleneck is in the database layer (N+1 queries). "
                "Authentication is solid but lacks rate limiting. "
                "Deployment pipeline is missing rollback automation."
            ),
            "db_finding": (
                "Database: N+1 query pattern detected in /api/orders endpoint. "
                "SELECT * FROM orders followed by per-row SELECT * FROM line_items. "
                "Fix: add eager loading with JOIN. Add index on orders(user_id, created_at)."
            ),
            "auth_finding": (
                "Auth: JWT tokens validated correctly. No secret key rotation policy. "
                "Recommendation: rotate JWT_SECRET every 90 days. "
                "Rate limiting missing — add 1000 req/min per authenticated user."
            ),
            "devops_finding": (
                "DevOps: CI passes but no automated rollback on error spike. "
                "Add: monitor error rate post-deploy, auto-rollback if > 1% in 5 min. "
                "Staging environment not gated — any engineer can deploy to prod."
            ),
        }

        # Persist each finding as a separate memory
        paths: dict[str, str] = {}
        for key, content in findings.items():
            path = self.remember(
                key,
                content,
                tags=["research", topic.replace(" ", "_"), key],
                layer="knowledge",
            )
            paths[key] = path
            print(f"  ✓ stored {path}")

        # Link the findings together in the knowledge graph
        try:
            self._mem.relate(
                paths["db_finding"],
                paths["summary"],
                relationship="supports",
                strength=0.9,
            )
            self._mem.relate(
                paths["auth_finding"],
                paths["summary"],
                relationship="supports",
                strength=0.9,
            )
        except Exception:
            pass  # relationships are optional extras

        return paths


# ── Writer agent ───────────────────────────────────────────────────────────────

class WriterAgent(BaseAgent):
    """Reads research findings from LLMFS and drafts recommendations."""

    def draft_report(self, topic: str, researcher_ns: str) -> str:
        """Search for research findings and generate a report."""
        print(f"\n[{self.name}] Drafting report on: {topic!r}")
        print(f"  Searching LLMFS (researcher namespace: {researcher_ns})")

        # ── Read findings written by the researcher ────────────────────────────
        results = self.recall(
            f"findings about {topic}",
            k=5,
            path_prefix=researcher_ns,
        )

        if not results:
            return "No research findings found. Has the researcher run yet?"

        print(f"  Found {len(results)} relevant memories:")
        for r in results:
            print(f"    [{r.score:.3f}] {r.path}")

        # Compile findings into a report (mock composition)
        report_sections: list[str] = [
            f"# Technical Report: {topic}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Executive Summary",
        ]
        for r in results[:1]:  # use top result as summary
            report_sections.append(r.chunk_text[:300])

        report_sections.append("\n## Detailed Findings")
        for r in results[1:]:
            report_sections.append(f"\n### From {r.path}")
            report_sections.append(r.chunk_text[:250])

        report_sections.append("\n## Recommendations")
        report_sections.append(
            "1. Fix N+1 queries in orders endpoint (immediate — high impact)\n"
            "2. Implement JWT secret rotation policy (30 days)\n"
            "3. Add rate limiting (1000 req/min authenticated)\n"
            "4. Automate rollback in CI/CD pipeline"
        )

        report = "\n".join(report_sections)

        # Persist the report under the writer's namespace
        report_path = self.remember(
            "final_report",
            report,
            tags=["report", topic.replace(" ", "_"), "final"],
            layer="knowledge",
        )
        print(f"  ✓ report stored at {report_path}")
        return report


# ── Executive agent ────────────────────────────────────────────────────────────

class ExecutiveAgent(BaseAgent):
    """Reads the final report and produces a terse action plan."""

    def create_action_plan(self, report_path: str) -> str:
        """Read a writer's report and distil into action items."""
        print(f"\n[{self.name}] Reading report at {report_path}")
        report_content = self.read_shared(report_path)

        # Search for any additional context across all agents
        extra = self.recall("critical security recommendations", k=2)
        print(f"  Additional context found: {len(extra)} memories")

        # Mock: extract action items (real agent would call LLM here)
        action_plan = (
            "ACTION PLAN (Sprint 1):\n"
            "  [P0] Fix N+1 queries — 2 days, @backend-team\n"
            "  [P1] Add rate limiting — 1 day, @platform-team\n"
            "  [P1] JWT secret rotation policy — 1 day, @security-team\n"
            "  [P2] CI/CD rollback automation — 3 days, @devops-team\n"
            "  [P2] Staging gate — 0.5 days, @devops-team\n"
        )

        action_path = self.remember(
            "sprint1_actions",
            action_plan,
            tags=["action-plan", "sprint1"],
            layer="knowledge",
        )
        print(f"  ✓ action plan stored at {action_path}")
        return action_plan


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_pipeline(mem_path: str) -> None:
    """Run the three-agent pipeline sharing a single MemoryFS."""

    mem = MemoryFS(path=mem_path)
    print(f"Shared MemoryFS at: {mem_path}")
    print("Three agents will collaborate on a codebase review.\n")

    TOPIC = "codebase quality review"

    # Instantiate agents — all sharing the SAME mem instance
    researcher = ResearcherAgent("Researcher", mem, namespace="/agents/researcher")
    writer = WriterAgent("Writer", mem, namespace="/agents/writer")
    executive = ExecutiveAgent("Executive", mem, namespace="/agents/executive")

    # ── Step 1: Researcher investigates ───────────────────────────────────────
    finding_paths = researcher.investigate(TOPIC)

    # ── Step 2: Writer reads findings and drafts report ────────────────────────
    report = writer.draft_report(TOPIC, researcher_ns="/agents/researcher")
    report_path = "/agents/writer/final_report"

    # ── Step 3: Executive turns report into an action plan ─────────────────────
    action_plan = executive.create_action_plan(report_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nAction Plan (from Executive):")
    print(textwrap.indent(action_plan, "  "))

    status = mem.status()
    print(f"\nShared MemoryFS totals:")
    print(f"  Memories: {status['total']}")
    print(f"  Layers  : {status['layers']}")
    print(f"  Disk    : {status['disk_mb']:.3f} MB")

    # Show all memories across all agents
    all_memories = mem.list("/agents")
    print(f"\nAll agent memories ({len(all_memories)}):")
    for obj in sorted(all_memories, key=lambda o: o.path):
        print(f"  {obj.path}  [{obj.layer}]  tags={obj.tags}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llmfs_multiagent_") as tmp:
        run_pipeline(tmp)
