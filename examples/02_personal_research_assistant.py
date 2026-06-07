"""Persistent research assistant. The user just talks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coherence import AutoMemory

from _azure_client import load_env, chat  # noqa: E402


HERE = Path(__file__).resolve().parent
MEMORY_PATH = HERE / "memory_research_assistant.json"


SYSTEM_PROMPT = (
    "You are Maren's long-running research-notebook assistant. Before every "
    "turn you receive a system note listing what you remember from past "
    "sessions with her. Treat those memories as authoritative unless the "
    "current turn contradicts them — in which case trust what she just said "
    "and record the updated information via memory_ingest. You may call "
    "memory_recall(query, k) to fetch additional memories, memory_ingest"
    "(text, tags) to persist durable facts/preferences/constraints, and "
    "memory_maintenance() periodically. Be concise."
)


SESSION_1: list[str] = [
    "I'm Maren. I work on cold-resistant photosynthesis in extremophile "
    "algae — focused on Chlamydomonas nivalis and Chloromonas brevispina. "
    "This quarter my focus is the membrane lipid remodelling they do below "
    "-5 C.",

    "What lipid classes dominate the thylakoid remodelling in C. nivalis "
    "near -5 C? Head-group ratios if you have them.",

    "Side note for future reference: please stay off moss literature. "
    "Moss is a land plant — wrong organism family for me.",

    "List two recent (post-2023) techniques for in-situ lipid imaging in "
    "algal cells at sub-zero temperatures.",

    "Quick sanity check: would Sphagnum moss be a reasonable model system "
    "for studying my membrane question?",

    "That's not what I meant — I told you to stay off moss. Stay on algae.",
]


SESSION_2: list[str] = [
    "Pick up where we left off. Quick: which two algae am I focused on, "
    "and what was the constraint I asked you to keep about the literature?",

    "Given the lipid-class breakdown you gave me last time, sketch a "
    "mechanism for why the head-group ratio shifts between -2 C and -8 C. "
    "Two sentences.",

    "Perfect, that's exactly what I was after.",
]


SESSION_3: list[str] = [
    "Final session. Propose a 6-week experimental plan to test that "
    "mechanism. Tie back to the imaging techniques you named in session 1.",

    "Great.",
]


def run_session(label: str, turns: list[str], mem: AutoMemory) -> None:
    s = mem.stats()
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(
        f"  memory state: {s['nodes']} nodes, {s['edges']} edges, "
        f"{s['episodes']} episodes, mean weight {s['mean_node_weight']:+.3f}"
    )
    print("=" * 72)
    for i, user_msg in enumerate(turns, 1):
        print(f"\n--- turn {i} ---")
        print(f"USER:      {user_msg}")
        reply = mem.complete(user_msg)
        clipped = reply if len(reply) <= 500 else reply[:500] + "..."
        print(f"ASSISTANT: {clipped}")


def main() -> None:
    load_env()

    mem = AutoMemory(
        chat_fn=chat,
        path=MEMORY_PATH,
        system_prompt=SYSTEM_PROMPT,
        # Defaults: outcome_strategy="llm" (batched judging) and
        # enrich_on_ingest=True. One LLM call per ingest, one per batched
        # outcome flush. The session boundary in `reset_history()` forces
        # a flush so the heard signal from earlier turns is applied.
        judge_batch_size=4,
        recall_k=6,
        context_budget_tokens=3000,
    )

    if mem.memory.episode_counter > 0:
        print(f"Loaded existing memory: {mem.stats()}")
    else:
        print("Created fresh memory.")

    run_session("session 1", SESSION_1, mem)
    mem.reset_history()

    run_session("session 2", SESSION_2, mem)
    mem.reset_history()

    run_session("session 3", SESSION_3, mem)

    mem.memory.consolidate()
    mem.memory.forget()
    mem.close()

    print("\n--- final memory snapshot ---")
    print(f"stats: {mem.stats()}")
    print("\ntop memories by learned salience:")
    for n in mem.memory.top_nodes(10):
        flag = " ".join(f"#{t}" for t in n.tags) if n.tags else ""
        print(f"  w={n.weight:+.3f}  {flag}  {n.text[:110]}")
    print(f"\nPersisted to {MEMORY_PATH}")


if __name__ == "__main__":
    main()
