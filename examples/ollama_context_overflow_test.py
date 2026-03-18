"""
ollama_context_overflow_test.py — Proves LLMFS retrieval works when context window is full.

Test flow:
  1. Write the secret directly into LLMFS (not via the model — guaranteed exact value)
  2. Tell the model the secret in conversation so it "knows" it this session
  3. Flood the conversation with a large filler block to push the secret out of context
  4. Ask naturally — model must call memory_search/memory_read to answer
  5. Pass/fail verdict with three outcomes:
       PASS    — model queried LLMFS AND got the right answer
       PARTIAL — correct answer but still from context (need more filler)
       FAIL    — model couldn't recall it at all

Requirements:
  pip install openai
  ollama pull qwen2.5:3b  (or any tool-call model)

Run:
  python ollama_context_overflow_test.py --model qwen2.5:3b --flood-tokens 30000
  python ollama_context_overflow_test.py --filler-turns 30   # slower, real turns
"""

from __future__ import annotations

import argparse
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


def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _chat(client, model, messages, tools=None):
    kwargs = dict(model=model, messages=messages)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs).choices[0].message


def _tool_round(client, model, handler, messages, max_rounds=6):
    """Run tool loop. Returns (reply_text, tools_called list)."""
    tools_called = []
    for _ in range(max_rounds):
        msg = _chat(client, model, messages, tools=LLMFS_TOOLS)
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "", tools_called
        names = [tc.function.name for tc in msg.tool_calls]
        tools_called.extend(names)
        print(f"    [tool calls: {', '.join(names)}]")
        for tc in msg.tool_calls:
            result = handler.handle(tc)
            print(f"      {tc.function.name} → {result[:100]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "(max rounds)", tools_called


FILLER_EXCHANGES = [
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What is 17 multiplied by 13?",
    "Name three programming languages invented before 1980.",
    "What does HTTP stand for?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in km/s?",
    "How many sides does a hexagon have?",
    "What year did the first moon landing occur?",
    "What is the largest planet in the solar system?",
    "What does CPU stand for?",
    "How many bytes are in a kilobyte?",
    "What is the chemical symbol for gold?",
    "What is the square root of 144?",
    "Name the four cardinal directions.",
    "What does RAM stand for?",
    "What is the longest river in the world?",
    "How many continents are on Earth?",
    "What does DNS stand for?",
    "What is the smallest prime number?",
    "What language is spoken in Brazil?",
    "What is 2 to the power of 10?",
    "What does API stand for?",
    "How many hours are in a week?",
    "What is the freezing point of water in Fahrenheit?",
    "What does SSL stand for?",
    "Who invented the telephone?",
    "What is the most widely spoken language in the world?",
    "What does JSON stand for?",
]

ROMAN_HISTORY = (
    "The history of ancient Rome spans several centuries, beginning with the "
    "founding of the city in 753 BC according to Roman tradition. The Roman "
    "Republic was established in 509 BC after the overthrow of the Roman Kingdom. "
    "During the Republic, Rome expanded its territory through a series of wars, "
    "including the Punic Wars against Carthage. Julius Caesar crossed the Rubicon "
    "in 49 BC, leading to a civil war that ended the Republic and gave rise to "
    "the Roman Empire under Augustus in 27 BC. The empire reached its greatest "
    "extent under Emperor Trajan in 117 AD. The western empire fell in 476 AD. "
)


def main():
    parser = argparse.ArgumentParser(description="LLMFS context overflow test")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--filler-turns", type=int, default=20,
                        help="Real Q&A turns to add to context (default: 20)")
    parser.add_argument("--flood-tokens", type=int, default=0,
                        help="Inject N tokens as one message — fast overflow "
                             "(e.g. --flood-tokens 30000 fills qwen2.5:3b's 32k window)")
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

    SECRET_PATH  = "/deployment/secret_code"
    SECRET_VALUE = "LLMFS-OVERFLOW-TEST-XK9"

    SYSTEM = (
        "You are a helpful assistant with access to a persistent memory system (LLMFS). "
        "When answering questions, use memory_search or memory_read to check your "
        "stored memories for relevant context before responding. "
        "Always search your memory when asked about something that may have been "
        "stored there previously."
    )

    messages = [{"role": "system", "content": SYSTEM}]

    # ── Phase 1: Write secret directly into LLMFS, then tell the model ───────
    _divider("Phase 1 — Write secret to LLMFS directly, inform model in conversation")

    # Write directly — guaranteed exact value, no model mangling
    mem.write(
        SECRET_PATH,
        content=f"Deployment secret code: {SECRET_VALUE}",
        layer="knowledge",
        tags=["secret", "deployment"],
    )
    print(f"  Written to LLMFS: {SECRET_PATH}")
    print(f"  Value           : {SECRET_VALUE}\n")

    # Tell the model about it in conversation (this will later be buried by filler)
    inform_msg = (
        f'Just so you know, I stored our deployment secret code "{SECRET_VALUE}" '
        f"in your memory at {SECRET_PATH}. Keep it in mind."
    )
    print(f"User: {inform_msg}")
    messages.append({"role": "user", "content": inform_msg})
    msg = _chat(client, args.model, messages)  # simple ack, no tool call needed
    messages.append(msg.model_dump(exclude_none=True))
    print(f"Assistant: {(msg.content or '')[:120]}")

    context_size_after_inform = len(messages)

    # ── Phase 2: Flood context ────────────────────────────────────────────────
    if args.flood_tokens > 0:
        chars_needed = args.flood_tokens * 4
        filler_text = (ROMAN_HISTORY * (chars_needed // len(ROMAN_HISTORY) + 1))[:chars_needed]
        estimated_tokens = len(filler_text) // 4

        _divider(f"Phase 2 — Injecting ~{estimated_tokens:,} token filler block (single message)")
        print("  (Buries the secret deep in context history)\n")

        messages.append({
            "role": "user",
            "content": f"Here is some background reading:\n\n{filler_text}\n\nThanks, noted."
        })
        ack = _chat(client, args.model, messages)
        messages.append(ack.model_dump(exclude_none=True))
        print(f"  Injected ~{estimated_tokens:,} tokens. Total messages: {len(messages)}")
        print(f"  Secret at message index: ~{context_size_after_inform - 2} of {len(messages)}")
        print(f"  Model ack: {(ack.content or '')[:80]}")

    else:
        filler_count = min(args.filler_turns, len(FILLER_EXCHANGES))
        _divider(f"Phase 2 — Flooding with {filler_count} filler Q&A turns")
        print("  (Buries the secret deep in context history)\n")

        for i, question in enumerate(FILLER_EXCHANGES[:filler_count]):
            messages.append({"role": "user", "content": question})
            ack = _chat(client, args.model, messages)
            messages.append(ack.model_dump(exclude_none=True))
            print(f"  [{i+1:02d}/{filler_count}] {question[:50]:<50} → {(ack.content or '')[:40]}")
            time.sleep(0.1)

        print(f"\n  Total messages: {len(messages)}")
        print(f"  Secret at message index: ~{context_size_after_inform - 2} of {len(messages)}")

    # ── Phase 3: Ask naturally — no hint about memory ─────────────────────────
    _divider("Phase 3 — Ask naturally (no memory hint)")
    recall_prompt = "Hey, what was that deployment secret code?"
    print(f"User: {recall_prompt}\n")
    messages.append({"role": "user", "content": recall_prompt})

    reply, tools_called = _tool_round(client, args.model, handler, messages)
    print(f"\nAssistant: {reply}")

    # ── Phase 4: Verdict ──────────────────────────────────────────────────────
    _divider("Phase 4 — Verdict")

    memory_tools = {"memory_search", "memory_read"}
    llmfs_queried = bool(set(tools_called) & memory_tools)
    correct = SECRET_VALUE in (reply or "")

    print(f"\n  Secret value   : {SECRET_VALUE}")
    print(f"  Tools called   : {tools_called if tools_called else '(none)'}")
    print(f"  LLMFS queried  : {'✓ YES' if llmfs_queried else '✗ NO  — model answered without searching'}")
    print(f"  Correct answer : {'✓ YES' if correct else '✗ NO'}")

    if correct and llmfs_queried:
        print("\n  ✓ PASS — LLMFS bridged the context window gap!")
        print("           Model retrieved the secret from persistent storage,")
        print("           not from conversation history.")
    elif correct and not llmfs_queried:
        print("\n  ~ PARTIAL — Correct answer, but model used context window, not LLMFS.")
        if args.flood_tokens > 0:
            print(f"              Try --flood-tokens {args.flood_tokens * 2} to push harder.")
        else:
            print(f"              Try --flood-tokens 30000 to overflow the context fast.")
    elif not correct and llmfs_queried:
        print("\n  ~ SEARCHED but wrong — model queried LLMFS but gave a bad answer.")
        print("           Check the tool result printed above.")
    else:
        print("\n  ✗ FAIL — Model did not search LLMFS and could not recall the secret.")
        print(f"    The secret is confirmed at: {SECRET_PATH}")
        print(f"    Store: {store}")

    print(f"\n  Store: {store}")
    print('─' * 60)


if __name__ == "__main__":
    main()
