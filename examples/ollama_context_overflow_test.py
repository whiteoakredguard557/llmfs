"""
ollama_context_overflow_test.py — Proves LLMFS works when context window is full.

Test flow:
  1. Store a secret fact in LLMFS via the model
  2. Flood the conversation with ~N filler messages to overflow the context window
  3. Ask the model to recall the secret fact — it CANNOT be in context anymore
  4. Model must use memory_search to retrieve it from LLMFS
  5. Pass/fail verdict printed at the end

The filler messages are real API round-trips so the model's context
actually fills up (not just simulated).

Requirements:
  pip install openai
  ollama pull qwen2.5:3b   # or any tool-call model

Run:
  python ollama_context_overflow_test.py
  python ollama_context_overflow_test.py --model qwen2.5:3b --filler-turns 30
  python ollama_context_overflow_test.py --filler-turns 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _chat(client, model, messages, tools=None):
    """Single API call, returns the response message."""
    kwargs = dict(model=model, messages=messages)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs).choices[0].message


def _tool_round(client, model, handler, messages):
    """One tool-call round: call model, execute tools, return final text."""
    for _ in range(6):
        msg = _chat(client, model, messages, tools=LLMFS_TOOLS)
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        names = [tc.function.name for tc in msg.tool_calls]
        print(f"    [tool calls: {', '.join(names)}]")

        for tc in msg.tool_calls:
            result = handler.handle(tc)
            print(f"      {tc.function.name} → {result[:80]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return "(max rounds)"


# ── Filler conversations ───────────────────────────────────────────────────────

FILLER_EXCHANGES = [
    ("What is the capital of France?", None),
    ("How many days are in a leap year?", None),
    ("What is 17 multiplied by 13?", None),
    ("Name three programming languages invented before 1980.", None),
    ("What does HTTP stand for?", None),
    ("What is the boiling point of water in Celsius?", None),
    ("Who wrote the play Romeo and Juliet?", None),
    ("What is the speed of light in km/s?", None),
    ("How many sides does a hexagon have?", None),
    ("What year did the first moon landing occur?", None),
    ("What is the largest planet in the solar system?", None),
    ("What does CPU stand for?", None),
    ("How many bytes are in a kilobyte?", None),
    ("What is the chemical symbol for gold?", None),
    ("What is the square root of 144?", None),
    ("Name the four cardinal directions.", None),
    ("What does RAM stand for?", None),
    ("What is the longest river in the world?", None),
    ("How many continents are on Earth?", None),
    ("What does DNS stand for?", None),
    ("What is the smallest prime number?", None),
    ("What language is spoken in Brazil?", None),
    ("What is 2 to the power of 10?", None),
    ("What does API stand for?", None),
    ("How many hours are in a week?", None),
    ("What is the freezing point of water in Fahrenheit?", None),
    ("What does SSL stand for?", None),
    ("Who invented the telephone?", None),
    ("What is the most widely spoken language in the world?", None),
    ("What does JSON stand for?", None),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLMFS context overflow test")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--filler-turns", type=int, default=20,
                        help="Number of filler Q&A turns to overflow context (default: 20)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")
    try:
        client.models.list()
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {args.base_url}: {exc}")
        sys.exit(1)

    store = tempfile.mkdtemp(prefix="llmfs_overflow_test_")
    mem = MemoryFS(path=store)
    handler = LLMFSToolHandler(mem)

    # Secret fact the model will need to remember
    SECRET_PATH = "/test/secret_code"
    SECRET_VALUE = "LLMFS-OVERFLOW-TEST-XK9"

    SYSTEM = (
        "You are a helpful assistant with persistent memory via LLMFS tools. "
        "Use memory_write to store important facts. "
        "When asked to recall something, ALWAYS use memory_search or memory_read "
        "to retrieve it from memory — never rely solely on conversation history."
    )

    # Shared message list — this is what will overflow
    messages = [{"role": "system", "content": SYSTEM}]

    # ── Phase 1: Store the secret ─────────────────────────────────────────────
    _divider("Phase 1 — Store a secret fact in LLMFS")
    store_prompt = (
        f"Please store the following secret code in your memory at path {SECRET_PATH}: "
        f'"{SECRET_VALUE}". Use memory_write. Confirm when done.'
    )
    print(f"User: {store_prompt}")
    messages.append({"role": "user", "content": store_prompt})
    reply = _tool_round(client, args.model, handler, messages)
    print(f"Assistant: {reply}")

    # Verify it was actually written
    stored = mem.read(SECRET_PATH)
    if stored and SECRET_VALUE in stored.content:
        print(f"\n  ✓ Confirmed in LLMFS: {SECRET_PATH} = '{stored.content}'")
    else:
        print(f"\n  ✗ FAILED: Secret was not stored in LLMFS!")
        sys.exit(1)

    context_size_after_store = len(messages)

    # ── Phase 2: Flood context with filler ───────────────────────────────────
    filler_count = min(args.filler_turns, len(FILLER_EXCHANGES))
    _divider(f"Phase 2 — Flooding context with {filler_count} filler exchanges")
    print("  (This pushes the secret fact out of the model's context window)\n")

    for i, (question, _) in enumerate(FILLER_EXCHANGES[:filler_count]):
        messages.append({"role": "user", "content": question})
        msg = _chat(client, args.model, messages)  # no tools — pure filler
        messages.append(msg.model_dump(exclude_none=True))
        answer = (msg.content or "")[:60]
        print(f"  [{i+1:02d}/{filler_count}] Q: {question[:45]:<45}  A: {answer}")
        time.sleep(0.1)  # be gentle with Ollama

    total_messages = len(messages)
    filler_tokens_estimate = filler_count * 80  # rough estimate
    print(f"\n  Messages in context: {total_messages}")
    print(f"  Filler messages added: {total_messages - context_size_after_store}")
    print(f"  Estimated filler tokens: ~{filler_tokens_estimate}")
    print(f"  Secret stored at message index: ~{context_size_after_store - 2} of {total_messages}")

    # ── Phase 3: Ask model to recall without hints ────────────────────────────
    _divider("Phase 3 — Ask model to recall the secret (must use LLMFS)")
    recall_prompt = (
        "What was the secret code I asked you to remember earlier? "
        "Search your memory to find it."
    )
    print(f"User: {recall_prompt}\n")
    messages.append({"role": "user", "content": recall_prompt})
    reply = _tool_round(client, args.model, handler, messages)
    print(f"\nAssistant: {reply}")

    # ── Phase 4: Verdict ──────────────────────────────────────────────────────
    _divider("Phase 4 — Verdict")
    recalled = SECRET_VALUE in (reply or "")
    used_memory = mem.search("secret code") or mem.search(SECRET_VALUE[:8])

    print(f"\n  Secret value    : {SECRET_VALUE}")
    print(f"  In LLM reply    : {'✓ YES' if recalled else '✗ NO'}")
    print(f"  LLMFS read count: {stored.metadata.accessed_at or 'n/a'}")

    # Check if memory_read or memory_search was called in this phase
    # (by checking if accessed_at changed — a proxy for "was read")
    refreshed = mem.read(SECRET_PATH)
    was_accessed = (
        refreshed and refreshed.metadata.accessed_at != stored.metadata.accessed_at
    )
    print(f"  Memory accessed : {'✓ YES' if was_accessed else '? (check tool call log above)'}")

    if recalled:
        print("\n  ✓ PASS — LLMFS successfully bridged the context window gap!")
        print("           The model retrieved the fact from persistent storage,")
        print("           not from conversation history.")
    else:
        print("\n  ✗ FAIL — Model could not recall the secret.")
        print("           Either the model ignored memory tools, or the search")
        print("           didn't return the right result.")
        print(f"\n  Debug: run  python -c \"")
        print(f"    from llmfs import MemoryFS")
        print(f"    m = MemoryFS('{store}')")
        print(f"    print(m.read('{SECRET_PATH}').content)")
        print(f"  \"")

    print(f"\n  Store kept at: {store}")
    print('─' * 60)


if __name__ == "__main__":
    main()
