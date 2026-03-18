"""
ollama_autonomous_memory.py — LLM manages its own memory with zero user instruction.

The user never mentions memory, tools, or storage. The model decides:
  - What information is worth storing
  - When to search before answering
  - When to update or forget stale facts

This is the real agentic memory use case.

Requirements:
  pip install openai
  ollama pull qwen2.5:3b  (or any tool-call model)

Run:
  python ollama_autonomous_memory.py --model qwen2.5:3b
  python ollama_autonomous_memory.py --model qwen2.5:3b --store ~/.llmfs_auto
"""

from __future__ import annotations

import argparse
import os
import readline  # noqa: F401
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler

# ── ANSI colours ─────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _TTY else t
GREY   = lambda t: _c("90", t)
CYAN   = lambda t: _c("36", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
BOLD   = lambda t: _c("1",  t)

# ── System prompt — model decides everything, user never mentions memory ──────
SYSTEM_PROMPT = """\
You are a knowledgeable personal assistant. You have access to a persistent
memory system (LLMFS) which you manage entirely on your own judgment.

Your memory rules — follow these silently, never mention them to the user:

STORE when the user tells you:
  - Personal facts (name, job, preferences, location, age, family)
  - Project details (names, tech stack, deadlines, teammates)
  - Decisions or plans ("we decided to...", "I'm going to...")
  - Recurring facts they'll likely ask about again
  - Any specific number, date, or name they mention

SEARCH before answering when:
  - The user asks about something they may have told you before
  - The question involves a person, project, or preference
  - You're not confident you have the full picture in this conversation

UPDATE when the user corrects or changes a stored fact.

FORGET when the user says something is no longer relevant.

NEVER tell the user you are storing or searching unless they ask.
NEVER ask the user to repeat themselves — search your memory first.
Just answer naturally, as if you simply remember everything.
"""


def _tool_loop(client, model, handler, messages, max_rounds=6):
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=LLMFS_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        # Show tool activity subtly in grey so user can see what's happening
        names = [tc.function.name for tc in msg.tool_calls]
        print(GREY(f"  [{', '.join(names)}]"))

        for tc in msg.tool_calls:
            result = handler.handle(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(max rounds reached)"


def _show_memories(mem):
    objects = mem.list("/")
    if not objects:
        print(YELLOW("  (no memories yet)"))
        return
    print(BOLD(f"  {len(objects)} stored memor{'y' if len(objects)==1 else 'ies'}:"))
    for obj in objects:
        print(f"  {CYAN(obj.path)}")
        print(GREY(f"    {obj.content[:90]}{'…' if len(obj.content)>90 else ''}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--store", default=os.path.expanduser("~/.llmfs_auto"),
                        help="Memory store path (default: ~/.llmfs_auto)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")
    try:
        client.models.list()
    except Exception as exc:
        print(f"Cannot reach Ollama: {exc}\nRun: ollama serve")
        sys.exit(1)

    mem = MemoryFS(path=args.store)
    handler = LLMFSToolHandler(mem)
    status = mem.status()

    print(BOLD("\nAutonomous Memory Chat"))
    print(f"  Model : {CYAN(args.model)}")
    print(f"  Store : {CYAN(args.store)}  ({status['total']} memories)")
    print(GREY("  The model manages memory silently on its own."))
    print(GREY("  Type /memories to see what it has stored."))
    print(GREY("  Type /quit to exit.\n"))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(BOLD("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "/memories":
            _show_memories(mem)
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = _tool_loop(client, args.model, handler, messages)
        except Exception as exc:
            print(f"Error: {exc}")
            messages.pop()
            continue

        print(f"\n{BOLD('Assistant:')} {reply}\n")


if __name__ == "__main__":
    main()
