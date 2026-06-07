"""Learning-dynamics simulation: no LLM, no network."""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

# Allow running the file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coherence import Memory


SEED = 7
random.seed(SEED)


# --------------------------------------------------------------------- world

# Eight query patterns. Each has a topical vocabulary and a set of "true"
# causal facts. Decoys are written to share the topical vocabulary but assert
# the wrong specifics, so the lexical matcher will always pull both in.

QUERY_TOPICS: list[dict] = [
    {
        "query": "When did the marine survey near Stradbroke Island record the rare ribbon eel?",
        "causal_facts": [
            "Stradbroke Island marine survey recorded the rare ribbon eel on 14 April 2023.",
            "Field log for Stradbroke 2023 lists the ribbon eel sighting on April 14.",
        ],
        "decoy_facts": [
            "Stradbroke Island marine survey recorded the rare ribbon eel on 02 May 2021.",
            "Stradbroke Island marine survey recorded the rare ribbon eel on 30 June 2019.",
            "Stradbroke Island marine survey recorded the rare ribbon eel in January 2025.",
        ],
    },
    {
        "query": "Which alloy did the Reykjavik geothermal team pick for the deep-bore casing?",
        "causal_facts": [
            "Reykjavik geothermal team selected nickel-iron alloy GX-44 for the deep-bore casing.",
            "Deep-bore casing material at Reykjavik geothermal site is the GX-44 nickel-iron alloy.",
        ],
        "decoy_facts": [
            "Reykjavik geothermal team rejected titanium alloy TX-19 for the deep-bore casing.",
            "Reykjavik geothermal team selected copper-tin alloy CT-7 for the surface piping.",
            "Reykjavik geothermal team selected chromium alloy CR-22 for the deep-bore casing.",
        ],
    },
    {
        "query": "What is the operating frequency of the Aurelia phased-array radar?",
        "causal_facts": [
            "Aurelia phased-array radar operates at 9.6 GHz with a 200 MHz bandwidth.",
            "Aurelia radar centre frequency is 9.6 GHz; documented in the engineering note AU-12.",
        ],
        "decoy_facts": [
            "Aurelia phased-array radar operates at 5.8 GHz with a 100 MHz bandwidth.",
            "Aurelia phased-array radar prototype tested briefly at 12.4 GHz before being abandoned.",
            "Aurelia phased-array radar operates at 9.6 MHz with a 200 MHz bandwidth.",
        ],
    },
    {
        "query": "Who chaired the Selene committee for the Q3 lunar regolith report?",
        "causal_facts": [
            "Dr Mei Tanaka chaired the Selene committee for the Q3 lunar regolith report.",
            "Q3 lunar regolith report attributes leadership to Selene chair Dr Mei Tanaka.",
        ],
        "decoy_facts": [
            "Dr Rafael Ortiz chaired the Selene committee for the Q1 lunar regolith report.",
            "Dr Anya Volkov chaired the Selene committee for the Q3 atmospheric report.",
            "Dr Mei Tanaka chaired the Aurora committee for the Q3 lunar regolith report.",
        ],
    },
    {
        "query": "Which port handles the bonded shipment of the Helix vaccine candidates?",
        "causal_facts": [
            "Helix vaccine candidates ship as bonded freight through the port of Tallinn.",
            "Bonded shipment of Helix vaccine candidates is routed via Tallinn port.",
        ],
        "decoy_facts": [
            "Helix vaccine candidates ship as bonded freight through the port of Gdansk.",
            "Helix vaccine candidates ship as standard freight through the port of Tallinn.",
            "Helix booster shipments are routed via Hamburg port under bonded status.",
        ],
    },
    {
        "query": "What is the maximum sustained altitude of the Aeon-B atmospheric balloon?",
        "causal_facts": [
            "Aeon-B atmospheric balloon sustains a maximum altitude of 38.4 kilometres.",
            "Field record for Aeon-B shows a sustained ceiling of 38.4 km in the 2024 campaign.",
        ],
        "decoy_facts": [
            "Aeon-B atmospheric balloon sustains a maximum altitude of 28.4 kilometres.",
            "Aeon-A atmospheric balloon sustains a maximum altitude of 38.4 kilometres.",
            "Aeon-B atmospheric balloon has a brief peak of 41.0 km but not sustained.",
        ],
    },
    {
        "query": "What programming dialect runs the Cassini-2 onboard controllers?",
        "causal_facts": [
            "Cassini-2 onboard controllers run a hardened Ada-2012 dialect with custom Ravenscar profile.",
            "Onboard controller code for Cassini-2 is written in Ada-2012 under a custom Ravenscar profile.",
        ],
        "decoy_facts": [
            "Cassini-2 onboard controllers run a hardened C dialect with MISRA constraints.",
            "Cassini-2 ground station tooling is written in Python on a Linux build host.",
            "Cassini-1 onboard controllers ran a Pascal-derived dialect with custom profile.",
        ],
    },
    {
        "query": "Which catalyst accelerates the Helix triterpene cyclisation step?",
        "causal_facts": [
            "Scandium triflate catalyses the Helix triterpene cyclisation step with 86% yield.",
            "Helix triterpene cyclisation uses scandium triflate as the accelerating catalyst.",
        ],
        "decoy_facts": [
            "Iron triflate catalyses the Helix triterpene cyclisation step with 41% yield.",
            "Scandium triflate catalyses the Helix benzofuran rearrangement step with 86% yield.",
            "Scandium chloride catalyses the Helix triterpene cyclisation step with 12% yield.",
        ],
    },
]


NOISE_FACTS: list[str] = [
    "A volunteer cleanup of the kelp forest near Point Reyes removed 320 kg of derelict gear in June 2022.",
    "The municipal archive in Bergen restored a 1763 ship log of the Hanseatic trading fleet.",
    "Synthetic indigo dye prices fell 18% on the Mumbai spot market in the third quarter of 2024.",
    "A short film by Lina Boroni was awarded a special mention at the Sarajevo Film Festival in 2023.",
    "The Pacific Crest Trail Association revised its long-term trail maintenance guidelines in 2025.",
    "Composer Hisao Yamada released a string quartet titled Drift in late 2024.",
    "Civic broadband cooperatives in rural Quebec reached 41,000 households by the end of 2024.",
    "The Cambodian National Museum announced a temporary exhibit on Khmer bronze casting in 2026.",
    "A long-running citizen-science bee survey in Slovenia counted 612 species in its tenth year.",
    "The municipal library in Asuncion digitized 14,000 19th-century broadsheets between 2022 and 2025.",
    "Glassmakers on the island of Murano apprenticed 27 new artisans in 2026.",
    "A small mineral-water bottling plant opened in Kruja, Albania, employing 60 workers.",
]


def causal_texts_by_query() -> dict[int, set[str]]:
    return {q_idx: set(topic["causal_facts"]) for q_idx, topic in enumerate(QUERY_TOPICS)}


def populate(mem: Memory, *, init_jitter: float = 0.0, rng: random.Random | None = None) -> None:
    rng = rng or random.Random(0)

    def _add(text: str, tags: list[str]) -> None:
        jitter = rng.uniform(-init_jitter, init_jitter) if init_jitter else 0.0
        mem.ingest(text, tags=tags, initial_weight=mem.initial_weight + jitter)

    for q_idx, topic in enumerate(QUERY_TOPICS):
        for fact in topic["causal_facts"]:
            _add(fact, ["causal", f"q{q_idx}"])
        for fact in topic["decoy_facts"]:
            _add(fact, ["decoy", f"q{q_idx}"])
    # Noise: off-topic facts. Never picked, never reinforced; they decay
    # below the prune floor and get forgotten.
    for fact in NOISE_FACTS:
        _add(fact, ["noise"])


def causal_vs_decoy_stats(mem: Memory) -> dict[str, float]:
    causal = [n.weight for n in mem.nodes.values() if "causal" in n.tags]
    decoys = [n.weight for n in mem.nodes.values() if "decoy" in n.tags]
    return {
        "causal_mean": statistics.mean(causal) if causal else 0.0,
        "decoy_mean": statistics.mean(decoys) if decoys else 0.0,
        "causal_count": len(causal),
        "decoy_count": len(decoys),
    }


def causal_share_in_topk(mem: Memory, k: int = 3) -> float:
    causal_set = causal_texts_by_query()
    shares: list[float] = []
    for q_idx, topic in enumerate(QUERY_TOPICS):
        retrieved = mem.recall(topic["query"], k=k)
        if not retrieved:
            continue
        causal_count = sum(1 for n in retrieved if n.text in causal_set[q_idx])
        shares.append(causal_count / len(retrieved))
    return statistics.mean(shares) if shares else 0.0


def render_bar(value: float, width: int = 32, lo: float = 0.0, hi: float = 1.0) -> str:
    if hi <= lo:
        return ""
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(pct * width))
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    print("\n=== coherence example 01: learning-dynamics simulation ===\n")

    # Two memories with identical initial state: one learns from outcomes,
    # the other never gets reinforced (no-learning lexical baseline).
    trained = Memory(
        eta=0.30,
        eta_edge=0.04,
        decay_node=0.06,
        decay_edge=0.04,
        prune_floor=0.05,
        edge_floor=0.02,
        gamma=0.15,
        initial_weight=0.10,
        k_default=5,
        eligibility_kind="softmax",
        eligibility_temperature=0.15,
    )
    baseline = Memory(
        eta=0.0,             # no learning
        eta_edge=0.0,
        decay_node=0.0,
        decay_edge=0.0,
        prune_floor=0.0,
        edge_floor=0.0,
        gamma=0.0,           # lexical-only recall
        initial_weight=0.10,
        k_default=5,
    )

    init_rng_t = random.Random(SEED)
    init_rng_b = random.Random(SEED)
    populate(trained, init_jitter=0.02, rng=init_rng_t)
    populate(baseline, init_jitter=0.02, rng=init_rng_b)

    n_causal = sum(1 for n in trained.nodes.values() if "causal" in n.tags)
    n_decoys = sum(1 for n in trained.nodes.values() if "decoy" in n.tags)
    n_noise = sum(1 for n in trained.nodes.values() if "noise" in n.tags)
    print(
        f"Seeded memory with {len(trained.nodes)} nodes "
        f"({n_causal} causal, {n_decoys} decoys, {n_noise} noise)\n"
    )

    EPISODES = 300
    REPORT_EVERY = 30
    PROBE_K = 1
    EPSILON = 0.25            # exploration rate; decays with episode count

    print(
        f"Running {EPISODES} episodes (epsilon-greedy with eps={EPSILON} decaying linearly), "
        f"maintenance pass every {REPORT_EVERY}.\n"
        f"Probe metric: top-{PROBE_K} hit rate — is the agent's first pick "
        f"a causal fact? — across\n"
        f"{len(QUERY_TOPICS)} held-out queries (random baseline ~40%).\n"
    )
    print(
        f"{'episode':>8}  {'causal w':>9}  {'decoy w':>9}  "
        f"{'gap':>7}  {'nodes':>5}  {'trained hit':>11}  {'baseline hit':>12}  bar"
    )

    rng = random.Random(SEED + 1)
    causal_text_lookup = causal_texts_by_query()
    n_correct_recent = 0
    n_recent = 0

    for episode in range(1, EPISODES + 1):
        q_idx = rng.randrange(len(QUERY_TOPICS))
        topic = QUERY_TOPICS[q_idx]
        query = topic["query"]

        recalled = trained.recall(query, k=5)
        if not recalled:
            continue

        # Epsilon-greedy: exploit top-1 most of the time, sometimes try a
        # random one so the framework gets signal about decoys too.
        eps = EPSILON * (1.0 - episode / EPISODES)
        if rng.random() < eps:
            pick = rng.choice(recalled)
        else:
            pick = recalled[0]

        is_causal = pick.text in causal_text_lookup[q_idx]
        outcome = 1.0 if is_causal else -1.0
        trained.reinforce(query, [pick], outcome)
        n_correct_recent += int(is_causal)
        n_recent += 1

        if episode % REPORT_EVERY == 0:
            trained.forget()
            stats = causal_vs_decoy_stats(trained)
            gap = stats["causal_mean"] - stats["decoy_mean"]
            t_share = causal_share_in_topk(trained, k=PROBE_K)
            b_share = causal_share_in_topk(baseline, k=PROBE_K)
            recent_acc = n_correct_recent / max(n_recent, 1)
            print(
                f"{episode:>8}  {stats['causal_mean']:>9.3f}  "
                f"{stats['decoy_mean']:>9.3f}  {gap:>7.3f}  "
                f"{len(trained.nodes):>5}  {t_share:>11.1%}  {b_share:>12.1%}  "
                f"{render_bar(recent_acc)}"
            )
            n_correct_recent = 0
            n_recent = 0

    print("\n--- final state ---")
    final_stats = causal_vs_decoy_stats(trained)
    final_noise = sum(1 for n in trained.nodes.values() if "noise" in n.tags)
    print(
        f"causal retained: {final_stats['causal_count']}/{n_causal}, "
        f"decoy retained: {final_stats['decoy_count']}/{n_decoys}, "
        f"noise retained: {final_noise}/{n_noise}"
    )
    final_trained = causal_share_in_topk(trained, k=PROBE_K)
    final_baseline = causal_share_in_topk(baseline, k=PROBE_K)
    print(
        f"trained top-{PROBE_K} hit rate: {final_trained:.1%}    "
        f"baseline (no learning, no spread): {final_baseline:.1%}    "
        f"absolute gain: {final_trained - final_baseline:+.1%}"
    )

    edges_by_strength = sorted(trained.edges.items(), key=lambda kv: -abs(kv[1]))[:5]
    if edges_by_strength:
        print("\ntop learned associations (Hebbian edges):")
        for (i, j), w in edges_by_strength:
            ti = trained.nodes[i].text[:48].replace("\n", " ")
            tj = trained.nodes[j].text[:48].replace("\n", " ")
            print(f"  w={w:+.3f}  {ti!r}  <->  {tj!r}")

    # Save to disk so example 02 can pick it up if desired.
    out = Path(__file__).resolve().parent / "memory_simulation.json"
    trained.save(out)
    print(f"\nSaved trained memory to {out}")


if __name__ == "__main__":
    main()
