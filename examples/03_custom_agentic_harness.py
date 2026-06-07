"""Custom agentic harness with a verifiable QA bench."""

from __future__ import annotations

import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coherence import AutoMemory

from _azure_client import load_env, chat, first_message  # noqa: E402


HERE = Path(__file__).resolve().parent
MEMORY_PATH = HERE / "memory_custom_harness.json"


# ------------------------------------------------------------------- domain

GROUND_TRUTH_FACTS: list[str] = [
    "Project Lyra was founded on 14 September 2028 in Tromso, Norway, as a joint Nordic-Pacific research consortium.",
    "Project Lyra's first director is Dr. Anya Volkov, formerly chief biophysicist at the Reykjavik geothermal program.",
    "Project Lyra focuses on engineering cold-resistant photosynthesis in extremophile algae for closed-loop arctic agriculture.",
    "Project Lyra's lead chemistry lab uses scandium triflate as the catalyst for the triterpene cyclisation step, achieving 86% yield.",
    "Project Lyra's deep-bore casing material at the Reykjavik geothermal site is the GX-44 nickel-iron alloy.",
    "Project Lyra's Aurelia phased-array radar, used for atmospheric monitoring at the Tromso site, operates at 9.6 GHz with a 200 MHz bandwidth.",
    "Project Lyra's Aeon-B atmospheric balloon, used for stratospheric sampling, sustains a maximum altitude of 38.4 kilometres.",
    "Project Lyra's Cassini-2 onboard controllers run a hardened Ada-2012 dialect with a custom Ravenscar profile.",
    "Project Lyra's bonded vaccine-candidate shipments to its Baltic biocontainment partner are routed via the port of Tallinn.",
    "Project Lyra's Selene committee, chaired by Dr Mei Tanaka, published the Q3 lunar regolith report on 02 October 2029.",
    "Project Lyra's primary funding source for the 2029 fiscal year is the Vega Foundation, providing 64% of the operational budget.",
    "Project Lyra's marine survey near Stradbroke Island recorded the rare ribbon eel sighting on 14 April 2029.",
    "Project Lyra's IT stack mandates Postgres 17 with the TimescaleDB extension for all telemetry storage.",
    "Project Lyra's safety protocol GZ-7 requires double-redundant cold-chain breaks for any sample held below -18 C.",
    "Project Lyra's flagship publication, 'Cold Carbon Cycles' (2030), is co-authored by Volkov, Tanaka, and a 14-person team.",
]


DECOY_FACTS: list[str] = [
    "Project Lyra was founded in 2019 in Oslo, Norway, as a purely Nordic research consortium.",
    "Project Lyra's first director is Dr. Karim Hassan, who previously led the Aurora atmospheric programme.",
    "Project Lyra focuses on satellite-based remote sensing of arctic ice-sheet thickness using L-band synthetic-aperture radar.",
    "Project Lyra's lead chemistry lab uses iron triflate as the catalyst for the triterpene cyclisation step, achieving 41% yield.",
    "Project Lyra's deep-bore casing material at the Reykjavik geothermal site is the chromium alloy CR-22.",
    "Project Lyra's Aurelia phased-array radar operates at 5.8 GHz with a 100 MHz bandwidth, optimised for short-range tracking.",
    "Project Lyra's Aeon-A atmospheric balloon sustains a maximum altitude of 38.4 kilometres; Aeon-B was retired in 2024.",
    "Project Lyra's Cassini-2 onboard controllers run a hardened C dialect under MISRA-C constraints.",
    "Project Lyra's bonded vaccine-candidate shipments to its Baltic biocontainment partner are routed via the port of Gdansk.",
    "Project Lyra's Selene committee, chaired by Dr Rafael Ortiz, published the Q1 lunar regolith report in early 2029.",
    "Project Lyra's primary funding source for the 2029 fiscal year is the Nexus Foundation, providing 78% of the operational budget.",
    "Project Lyra's marine survey near Stradbroke Island recorded the rare ribbon eel sighting on 02 May 2027.",
    "Project Lyra's IT stack mandates ClickHouse for all telemetry storage; Postgres is used only for billing.",
    "Project Lyra's safety protocol GZ-7 requires triple-redundant cold-chain breaks for any sample held below -25 C.",
    "Project Lyra's flagship publication, 'Cold Carbon Cycles' (2030), is single-authored by Volkov.",
]


QUESTIONS: list[dict[str, Any]] = [
    {
        "q": "On what date was Project Lyra founded, and in which city?",
        "options": {
            "A": "14 September 2028 in Tromso, Norway",
            "B": "02 May 2019 in Oslo, Norway",
            "C": "01 January 2030 in Reykjavik, Iceland",
            "D": "14 April 2029 in Stradbroke, Australia",
        },
        "correct": "A",
    },
    {
        "q": "Who is Project Lyra's first director?",
        "options": {
            "A": "Dr Karim Hassan",
            "B": "Dr Anya Volkov",
            "C": "Dr Mei Tanaka",
            "D": "Dr Rafael Ortiz",
        },
        "correct": "B",
    },
    {
        "q": "What is the primary research focus of Project Lyra?",
        "options": {
            "A": "Satellite-based remote sensing of arctic ice",
            "B": "Engineering cold-resistant photosynthesis in extremophile algae",
            "C": "Manned exploration of the lunar south pole",
            "D": "Bonded biocontainment logistics across the Baltic",
        },
        "correct": "B",
    },
    {
        "q": "Which catalyst does Project Lyra's lead chemistry lab use for the triterpene cyclisation step?",
        "options": {
            "A": "Iron triflate, 41% yield",
            "B": "Scandium triflate, 86% yield",
            "C": "Scandium chloride, 12% yield",
            "D": "Palladium acetate, 73% yield",
        },
        "correct": "B",
    },
    {
        "q": "What casing material does Project Lyra use for the Reykjavik deep-bore?",
        "options": {
            "A": "GX-44 nickel-iron alloy",
            "B": "TX-19 titanium alloy",
            "C": "CR-22 chromium alloy",
            "D": "CT-7 copper-tin alloy",
        },
        "correct": "A",
    },
    {
        "q": "What is the operating frequency of the Aurelia phased-array radar?",
        "options": {
            "A": "5.8 GHz, 100 MHz bandwidth",
            "B": "9.6 GHz, 200 MHz bandwidth",
            "C": "12.4 GHz, 50 MHz bandwidth",
            "D": "9.6 MHz, 200 MHz bandwidth",
        },
        "correct": "B",
    },
    {
        "q": "What is the maximum sustained altitude of the Aeon-B balloon?",
        "options": {
            "A": "28.4 km",
            "B": "38.4 km",
            "C": "41.0 km",
            "D": "Aeon-B was retired before reaching its target altitude",
        },
        "correct": "B",
    },
    {
        "q": "What programming dialect runs the Cassini-2 onboard controllers?",
        "options": {
            "A": "MISRA-C",
            "B": "Hardened Ada-2012 with a custom Ravenscar profile",
            "C": "Python on a Linux build host",
            "D": "Pascal with a custom profile",
        },
        "correct": "B",
    },
    {
        "q": "Which port routes Project Lyra's bonded vaccine-candidate shipments to the Baltic partner?",
        "options": {"A": "Tallinn", "B": "Gdansk", "C": "Hamburg", "D": "Riga"},
        "correct": "A",
    },
    {
        "q": "Who chaired the Selene committee for the Q3 lunar regolith report?",
        "options": {
            "A": "Dr Rafael Ortiz",
            "B": "Dr Anya Volkov",
            "C": "Dr Mei Tanaka",
            "D": "Dr Karim Hassan",
        },
        "correct": "C",
    },
    {
        "q": "What is Project Lyra's primary 2029 funding source?",
        "options": {
            "A": "The Nexus Foundation, 78%",
            "B": "The Vega Foundation, 64%",
            "C": "The Aurora Foundation, 64%",
            "D": "The Vega Foundation, 78%",
        },
        "correct": "B",
    },
    {
        "q": "When was the Stradbroke-Island ribbon-eel sighting recorded?",
        "options": {
            "A": "02 May 2027",
            "B": "30 June 2019",
            "C": "14 April 2029",
            "D": "January 2025",
        },
        "correct": "C",
    },
    {
        "q": "Which database stack is mandated for Project Lyra telemetry?",
        "options": {
            "A": "ClickHouse only",
            "B": "Postgres 17 with the TimescaleDB extension",
            "C": "SQLite with a custom partitioner",
            "D": "Cassandra clustered across Tromso and Tallinn",
        },
        "correct": "B",
    },
    {
        "q": "What does safety protocol GZ-7 require?",
        "options": {
            "A": "Triple-redundant cold-chain breaks below -25 C",
            "B": "Double-redundant cold-chain breaks below -18 C",
            "C": "Single cold-chain break below -10 C",
            "D": "No cold-chain breaks; samples must remain frozen end to end",
        },
        "correct": "B",
    },
    {
        "q": "Who authored the 'Cold Carbon Cycles' (2030) flagship paper?",
        "options": {
            "A": "Single-authored by Volkov",
            "B": "Co-authored by Volkov and Tanaka only",
            "C": "Co-authored by Volkov, Tanaka, and a 14-person team",
            "D": "Single-authored by Tanaka",
        },
        "correct": "C",
    },
]


SYSTEM_PROMPT = (
    "You answer multiple-choice questions about the fictional Project Lyra "
    "consortium. Before every question you receive a system note listing "
    "memory excerpts retrieved from a knowledge graph. Some are correct, "
    "some are plausible decoys that contradict the correct ones. Decide "
    "using only the question wording and the relative consistency of the "
    "memories. Answer with exactly one letter A, B, C, or D, followed by a "
    "single short sentence of justification. Do not call any tool."
)


def format_question(question: dict[str, Any]) -> str:
    options = "\n".join(f"  {k}. {v}" for k, v in question["options"].items())
    return (
        f"Question: {question['q']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer (letter then one sentence):"
    )


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    if not text:
        return None
    m = _LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def run_epoch(
    mem: AutoMemory,
    epoch: int,
    bench: list[dict[str, Any]],
    truth_ids: set[str],
    decoy_ids: set[str],
) -> dict[str, Any]:
    n_correct = 0
    n_total = 0
    for q_idx, question in enumerate(bench, start=1):
        mem.reset_history()
        try:
            reply = mem.complete(format_question(question))
        except Exception as exc:  # noqa: BLE001
            print(f"  [epoch {epoch} q{q_idx}] chat error: {exc}")
            continue
        letter = parse_letter(reply)
        correct = letter == question["correct"]
        mem.report(+1.0 if correct else -1.0)
        n_correct += int(correct)
        n_total += 1
        marker = "OK" if correct else "XX"
        print(
            f"  [epoch {epoch} q{q_idx:>2}] {marker}  picked {letter or '?'}"
        )

    truth_weights = [
        mem.memory.nodes[nid].weight
        for nid in truth_ids
        if nid in mem.memory.nodes
    ]
    decoy_weights = [
        mem.memory.nodes[nid].weight
        for nid in decoy_ids
        if nid in mem.memory.nodes
    ]
    return {
        "epoch": epoch,
        "accuracy": n_correct / max(n_total, 1),
        "nodes": len(mem.memory.nodes),
        "truth_alive": len(truth_weights),
        "decoy_alive": len(decoy_weights),
        "truth_mean_w": statistics.mean(truth_weights) if truth_weights else 0.0,
        "decoy_mean_w": statistics.mean(decoy_weights) if decoy_weights else 0.0,
    }


def render_report(reports: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(
        f"{'epoch':>5}  {'accuracy':>8}  {'nodes':>5}  "
        f"{'truths':>6}  {'decoys':>6}  {'truth w':>8}  {'decoy w':>8}  {'gap':>6}"
    )
    for r in reports:
        gap = r["truth_mean_w"] - r["decoy_mean_w"]
        print(
            f"{r['epoch']:>5}  {r['accuracy']:>8.1%}  {r['nodes']:>5}  "
            f"{r['truth_alive']:>6}  {r['decoy_alive']:>6}  "
            f"{r['truth_mean_w']:>+8.3f}  {r['decoy_mean_w']:>+8.3f}  {gap:>+6.3f}"
        )
    print("=" * 80)


def main() -> None:
    load_env()

    mem = AutoMemory(
        chat_fn=chat,
        path=MEMORY_PATH,
        system_prompt=SYSTEM_PROMPT,
        outcome_strategy="manual",   # bench has ground truth — use it directly
        enrich_on_ingest=False,       # bench seeds 30 facts at once; skip per-ingest LLM
        expose_tools=False,
        recall_k=5,
        context_budget_tokens=1500,
        save_every_turn=False,
    )

    # Reset on each run so dynamics are reproducible.
    mem.memory.nodes.clear()
    mem.memory.edges.clear()
    mem.memory.index = type(mem.memory.index)()
    mem.memory.experiences.clear()
    mem.memory.episode_counter = 0

    # Randomize seed order so insertion ordering doesn't bias.
    rng = random.Random(13)
    pool: list[tuple[str, str]] = (
        [(t, "truth") for t in GROUND_TRUTH_FACTS]
        + [(t, "decoy") for t in DECOY_FACTS]
    )
    rng.shuffle(pool)
    truth_ids: set[str] = set()
    decoy_ids: set[str] = set()
    for text, kind in pool:
        nid = mem.remember(text, tags=[kind])
        (truth_ids if kind == "truth" else decoy_ids).add(nid)
    print(f"Seeded memory with {len(mem.memory.nodes)} nodes "
          f"({len(truth_ids)} truths, {len(decoy_ids)} decoys).")

    EPOCHS = 3
    epoch_rng = random.Random(29)
    bench = list(QUESTIONS)
    reports: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_rng.shuffle(bench)
        print(f"\n=== epoch {epoch} ===")
        r = run_epoch(mem, epoch, bench, truth_ids, decoy_ids)
        reports.append(r)
        if epoch < EPOCHS:
            forgotten = mem.memory.forget()
            removed = forgotten["removed_nodes"]
            pruned_t = sum(1 for nid in removed if nid in truth_ids)
            pruned_d = sum(1 for nid in removed if nid in decoy_ids)
            print(
                f"  [forget] pruned {len(removed)} nodes "
                f"({pruned_t} truths, {pruned_d} decoys)"
            )
        mem.save()

    elapsed = time.time() - started
    render_report(reports)
    accuracy_gain = reports[-1]["accuracy"] - reports[0]["accuracy"]
    weight_gap = reports[-1]["truth_mean_w"] - reports[-1]["decoy_mean_w"]
    print(
        f"\nFinished {len(reports)} epochs in {elapsed:.1f}s. "
        f"accuracy gain: {accuracy_gain:+.1%}, final truth-decoy gap: {weight_gap:+.3f}"
    )
    mem.close()
    print(f"Saved memory to {MEMORY_PATH}.")


if __name__ == "__main__":
    main()
