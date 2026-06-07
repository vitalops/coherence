from __future__ import annotations

import math

import pytest

from coherence import Memory


# --- ingest / recall --------------------------------------------------------


def test_ingest_and_recall_returns_lexical_hits():
    mem = Memory(eta=0.0, gamma=0.0, initial_weight=0.0)
    mem.ingest("Python is a programming language used widely.")
    mem.ingest("The ocean covers about seventy-one percent of Earth.")
    mem.ingest("Python was named after the Monty Python troupe.")

    hits = mem.recall("programming language Python", k=2)
    assert len(hits) == 2
    assert all("python" in n.text.lower() for n in hits)


def test_recall_handles_empty_memory():
    mem = Memory()
    assert mem.recall("anything", k=5) == []


def test_recall_handles_no_lexical_match():
    mem = Memory(initial_weight=0.3)
    mem.ingest("astrophysics")
    mem.ingest("biology")
    # Totally unrelated query: matcher returns zero, but the recall API
    # still returns the top-k by salience so the agent has something to use.
    hits = mem.recall("xyzzy", k=2)
    assert len(hits) == 2


# --- reinforcement dynamics -------------------------------------------------


def test_reinforce_increases_weight_on_success():
    mem = Memory(eta=0.5)
    nid = mem.ingest("the relevant fact about kettles")
    w_before = mem.nodes[nid].weight
    active = mem.recall("relevant fact kettle")
    mem.reinforce("relevant fact kettle", active, outcome=+1.0)
    assert mem.nodes[nid].weight > w_before


def test_reinforce_decreases_weight_on_failure():
    mem = Memory(eta=0.5)
    nid = mem.ingest("a fact")
    active = mem.recall("a fact")
    w_before = mem.nodes[nid].weight
    mem.reinforce("a fact", active, outcome=-1.0)
    assert mem.nodes[nid].weight < w_before


def test_repeated_success_separates_causal_from_decoy():
    mem = Memory(eta=0.4, eta_edge=0.0, gamma=0.0, initial_weight=0.1)
    causal = mem.ingest("Ada Lovelace wrote the first algorithm.")
    decoy = mem.ingest("Ada Lovelace ran the first algorithm tournament.")

    for _ in range(20):
        active = mem.recall("Ada Lovelace algorithm", k=2)
        # Success only when the causal node is the higher-activated one.
        if active and active[0].id == causal:
            mem.reinforce("Ada Lovelace algorithm", active, outcome=+1.0)
        else:
            mem.reinforce("Ada Lovelace algorithm", active, outcome=-1.0)

    assert mem.nodes[causal].weight > mem.nodes[decoy].weight


def test_eligibility_softmax_concentrates_credit():
    mem = Memory(
        eta=0.4,
        gamma=0.0,
        initial_weight=0.0,
        eligibility_kind="softmax",
        eligibility_temperature=0.05,
    )
    a = mem.ingest("alpha beta gamma delta")
    b = mem.ingest("epsilon zeta eta theta")
    active = mem.recall("alpha beta", k=2)
    mem.reinforce("alpha beta", active, outcome=+1.0)
    # The node that matched should grow much more than the also-active one.
    assert mem.nodes[a].weight > mem.nodes[b].weight + 0.05


# --- edges (Hebbian) --------------------------------------------------------


def test_hebbian_edges_form_on_co_activation():
    mem = Memory(eta=0.3, eta_edge=0.3, initial_weight=0.1, gamma=0.0)
    mem.ingest("red ripe apple from the orchard")
    mem.ingest("sweet orchard apple harvested in autumn")
    active = mem.recall("orchard apple", k=2)
    assert len(active) == 2
    mem.reinforce("orchard apple", active, outcome=+1.0)
    assert len(mem.edges) >= 1
    # Edge should be positive after a positive outcome.
    assert next(iter(mem.edges.values())) > 0


def test_edges_weaken_on_failure():
    mem = Memory(eta=0.3, eta_edge=0.5, initial_weight=0.2, gamma=0.0)
    mem.ingest("node alpha alpha alpha")
    mem.ingest("node beta beta beta")
    active = mem.recall("alpha beta", k=2)
    mem.reinforce("alpha beta", active, outcome=+1.0)
    w_pos = next(iter(mem.edges.values()))
    mem.reinforce("alpha beta", active, outcome=-1.0)
    w_neg = next(iter(mem.edges.values())) if mem.edges else 0.0
    assert w_neg < w_pos


# --- forget (decay + prune) -------------------------------------------------


def test_decay_shrinks_weight_toward_zero():
    mem = Memory(decay_node=0.2, prune_floor=0.0)
    nid = mem.ingest("x")
    mem.nodes[nid].weight = 1.0
    mem.forget()
    assert math.isclose(mem.nodes[nid].weight, 0.8, abs_tol=1e-9)


def test_prune_removes_below_floor():
    mem = Memory(decay_node=0.0, prune_floor=0.5)
    nid = mem.ingest("y", initial_weight=0.1)
    assert nid in mem.nodes
    mem.forget()
    assert nid not in mem.nodes


def test_prune_keeps_strong_node():
    mem = Memory(decay_node=0.0, prune_floor=0.5)
    nid = mem.ingest("z", initial_weight=1.0)
    mem.forget()
    assert nid in mem.nodes


def test_unbounded_growth_is_bounded_by_forget():
    mem = Memory(decay_node=0.3, prune_floor=0.05, initial_weight=0.1)
    for i in range(50):
        mem.ingest(f"transient fact number {i}")
    initial = len(mem.nodes)
    mem.forget()
    mem.forget()
    assert len(mem.nodes) < initial


# --- consolidate ------------------------------------------------------------


def test_consolidate_merges_near_duplicates():
    mem = Memory()
    a = mem.ingest("the cassowary is a large flightless bird from queensland")
    b = mem.ingest("the cassowary is a large flightless bird from queensland australia")
    mem.nodes[a].weight = 0.4
    mem.nodes[b].weight = 0.3
    merges = mem.consolidate(similarity_threshold=0.8)
    assert merges
    # Only one node should remain; its weight should be the sum.
    assert len(mem.nodes) == 1
    remaining = next(iter(mem.nodes.values()))
    assert math.isclose(remaining.weight, 0.7, abs_tol=1e-9)


# --- persistence ------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    mem = Memory(eta=0.3)
    a = mem.ingest("hello world")
    b = mem.ingest("goodbye world")
    active = mem.recall("world", k=2)
    mem.reinforce("world", active, outcome=+1.0)

    path = tmp_path / "memory.json"
    mem.save(path)

    loaded = Memory.load(path)
    assert set(loaded.nodes.keys()) == {a, b}
    assert loaded.episode_counter == mem.episode_counter
    # Edges should survive (one edge formed between the two co-active nodes).
    assert loaded.edges == mem.edges


def test_save_load_preserves_config(tmp_path):
    mem = Memory(eta=0.42, eta_edge=0.17, gamma=0.31, eligibility_kind="softmax")
    path = tmp_path / "config.json"
    mem.save(path)
    loaded = Memory.load(path)
    assert loaded.eta == 0.42
    assert loaded.eta_edge == 0.17
    assert loaded.gamma == 0.31
    assert loaded.eligibility_kind == "softmax"


# --- integration adapters ---------------------------------------------------


def test_openai_tool_specs_are_well_formed():
    from coherence.integrations import openai_tool_specs

    mem = Memory()
    specs = openai_tool_specs(mem)
    names = {s["function"]["name"] for s in specs}
    assert names == {
        "memory_recall",
        "memory_ingest",
        "memory_reinforce",
        "memory_maintenance",
    }
    for s in specs:
        # JSON Schema sanity
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)


def test_dispatch_round_trip():
    from coherence.integrations import dispatch

    mem = Memory()
    r = dispatch(mem, "memory_ingest", {"text": "first fact"})
    assert r["status"] == "ingested"
    nid = r["id"]

    recall = dispatch(mem, "memory_recall", {"query": "first fact", "k": 1})
    assert recall["memories"][0]["id"] == nid

    reinforce = dispatch(
        mem,
        "memory_reinforce",
        {"query": "first fact", "node_ids": [nid], "outcome": 1.0},
    )
    assert reinforce["status"] == "reinforced"
    assert mem.nodes[nid].weight > 0.1


def test_session_episode_context_manager():
    from coherence.integrations import MemorySession

    mem = Memory(eta=0.4)
    sess = MemorySession(mem)
    nid = sess.ingest("relevant insight")
    with sess.episode("relevant insight") as ep:
        assert any(n.id == nid for n in ep.nodes)
        ep.success()
    assert mem.nodes[nid].weight > 0.1


def test_session_episode_without_outcome_does_not_reinforce():
    from coherence.integrations import MemorySession

    mem = Memory(eta=0.4)
    nid = mem.ingest("fact")
    w_before = mem.nodes[nid].weight
    sess = MemorySession(mem)
    with sess.episode("fact"):
        pass  # no success / failure call
    assert mem.nodes[nid].weight == w_before


# --- anthropic protocol adapter ---------------------------------------------


def test_anthropic_tool_specs_shape():
    from coherence.integrations import anthropic_tool_specs

    mem = Memory()
    specs = anthropic_tool_specs(mem)
    for s in specs:
        assert "name" in s
        assert "description" in s
        assert s["input_schema"]["type"] == "object"


def test_anthropic_run_tool_call():
    from coherence.integrations import run_anthropic_tool_call

    mem = Memory()
    block = {
        "type": "tool_use",
        "id": "tu_1",
        "name": "memory_ingest",
        "input": {"text": "anthropic-bound fact"},
    }
    result = run_anthropic_tool_call(mem, block)
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tu_1"
    assert len(mem.nodes) == 1


# --- mcp adapter ------------------------------------------------------------


def test_mcp_tool_descriptors_handler_executes():
    from coherence.integrations import mcp_tool_descriptors

    mem = Memory()
    descs = mcp_tool_descriptors(mem)
    by_name = {d["name"]: d for d in descs}
    assert "memory_ingest" in by_name
    result = by_name["memory_ingest"]["handler"]({"text": "mcp-side fact"})
    assert result["status"] == "ingested"
    assert len(mem.nodes) == 1


# --- AutoMemory --------------------------------------------------------------


def _fake_chat_factory(replies=None, *, self_assess_score=0.0, recall_pick_first=0):
    """A fake chat_fn that routes by system prompt:

    - 'memory analyzer'           → return enrichment JSON (empty aliases)
    - 'evaluate an assistant'     → return ``self_assess_score`` per turn
    - 'memory retrieval engine'   → return first ``recall_pick_first`` IDs
    - anything else               → cycle through ``replies``
    """
    import re as _re
    seq = list(replies or [])
    state = {"i": 0}

    def fn(messages, tools=None, tool_choice=None, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        if "memory analyzer" in sys:
            return {"choices": [{"message": {"content": '{"aliases":[],"entities":[],"kind":"fact"}'}}]}
        if "evaluate an assistant" in sys:
            user = (messages[1].get("content", "") if len(messages) > 1 else "") or ""
            n_turns = user.count("=== Turn")
            lines = "\n".join(f"{i + 1}: {self_assess_score}" for i in range(n_turns))
            return {"choices": [{"message": {"content": lines}}]}
        if "memory retrieval engine" in sys:
            user = (messages[1].get("content", "") if len(messages) > 1 else "") or ""
            ids = _re.findall(r"\[([a-f0-9]{16})\]", user)
            take = ids[:recall_pick_first] if recall_pick_first else ids
            return {"choices": [{"message": {"content": "[" + ",".join(f'"{i}"' for i in take) + "]"}}]}
        content = seq[state["i"] % max(len(seq), 1)] if seq else "ok"
        state["i"] += 1
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    return fn


def test_automemory_complete_returns_reply():
    from coherence import AutoMemory

    mem = AutoMemory(chat_fn=_fake_chat_factory(["hello back"]), system_prompt="sys")
    reply = mem.complete("hi")
    assert reply == "hello back"


def test_automemory_self_assess_demotes_when_judge_negative():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["answer 1", "answer 2"], self_assess_score=-1.0),
        outcome_strategy="self_assess",
        judge_batch_size=1,
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("relevant fact")
    mem.complete("tell me the relevant fact")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("any follow-up")
    assert mem.memory.nodes[nid].weight < w_before


def test_automemory_self_assess_reinforces_when_judge_positive():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"], self_assess_score=+1.0),
        outcome_strategy="self_assess",
        judge_batch_size=1,
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("relevant fact")
    mem.complete("tell me the relevant fact")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("any follow-up")
    assert mem.memory.nodes[nid].weight > w_before


def test_automemory_self_assess_buffers_until_size():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["r1", "r2", "r3", "r4"], self_assess_score=+1.0),
        outcome_strategy="self_assess",
        judge_batch_size=3,
        intrinsic_retrieval_bump=0.0,
    )
    mem.remember("fact")
    mem.complete("q1")
    assert len(mem.memory.experiences) == 0  # nothing flushed yet
    mem.complete("q2")
    assert len(mem.memory.experiences) == 0
    mem.complete("q3")
    assert len(mem.memory.experiences) == 0
    mem.complete("q4")
    assert len(mem.memory.experiences) == 3


def test_automemory_close_flushes_remaining_buffer():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["x", "y"], self_assess_score=+1.0),
        outcome_strategy="self_assess",
        judge_batch_size=10,
        intrinsic_retrieval_bump=0.0,
    )
    mem.remember("fact")
    mem.complete("q1")
    mem.complete("q2")
    assert len(mem.memory.experiences) == 0
    mem.close()
    assert len(mem.memory.experiences) == 2


def test_self_assess_does_not_see_next_user_message():
    """The self-assess judge must NOT receive the next user message as evidence."""
    from coherence import AutoMemory

    captured_prompts: list[str] = []

    def chat(messages, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        if "evaluate an assistant" in sys:
            captured_prompts.append(messages[1]["content"])
            return {"choices": [{"message": {"content": "1: 0"}}]}
        if "memory analyzer" in sys:
            return {"choices": [{"message": {"content": '{"aliases":[],"entities":[],"kind":"fact"}'}}]}
        if "memory retrieval engine" in sys:
            return {"choices": [{"message": {"content": "[]"}}]}
        return {"choices": [{"message": {"content": "reply"}}]}

    mem = AutoMemory(chat_fn=chat, outcome_strategy="self_assess", judge_batch_size=1)
    mem.remember("a fact")
    mem.complete("first user message")
    mem.complete("SECOND USER MESSAGE that the judge should never see")
    assert captured_prompts, "self-assess judge was not called"
    # The next user message must not appear in the prompt the judge sees.
    assert "SECOND USER MESSAGE" not in captured_prompts[0]


def test_follow_up_fires_only_on_explicit_correction():
    from coherence.judging import follow_up_outcome

    # Explicit corrections → -1
    assert follow_up_outcome("q", "r", "that's not what I meant") == -1.0
    assert follow_up_outcome("q", "r", "no, that's wrong") == -1.0
    assert follow_up_outcome("q", "r", "let me clarify what I asked") == -1.0
    assert follow_up_outcome("q", "r", "I told you to stay off moss") == -1.0

    # Silence / politeness / smooth follow-ups → 0 (NO false positive)
    assert follow_up_outcome("q", "r", "thanks") == 0.0
    assert follow_up_outcome("q", "r", "perfect") == 0.0
    assert follow_up_outcome("q", "r", "What about the second part?") == 0.0
    assert follow_up_outcome("q", "r", "") == 0.0


def test_intrinsic_retrieval_bump_applies_each_turn():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["ok"]),
        outcome_strategy="manual",
        intrinsic_retrieval_bump=0.05,
    )
    nid = mem.remember("a relevant fact about cats")
    w0 = mem.memory.nodes[nid].weight
    mem.complete("cats?")
    w1 = mem.memory.nodes[nid].weight
    # The recalled node got +0.05 just for being retrieved.
    assert abs((w1 - w0) - 0.05) < 1e-6


def test_intrinsic_retrieval_bump_can_be_disabled():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["ok"]),
        outcome_strategy="manual",
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("fact")
    w0 = mem.memory.nodes[nid].weight
    mem.complete("query")
    assert mem.memory.nodes[nid].weight == w0


def test_manual_default_does_not_auto_infer():
    """With the new default strategy=manual, no LLM judging happens unless
    you ask for it. Disable intrinsic bump to isolate the outcome path."""
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"]),
        # No outcome_strategy specified → default "manual"
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("a fact")
    mem.complete("first")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("no, that's wrong — would have demoted under follow_up mode")
    # Default is manual; no auto-reinforcement.
    assert mem.memory.nodes[nid].weight == w_before


def test_follow_up_strategy_demotes_on_explicit_correction():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"]),
        outcome_strategy="follow_up",
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("a fact")
    mem.complete("tell me the fact")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("no, that's wrong")  # explicit correction
    assert mem.memory.nodes[nid].weight < w_before


def test_follow_up_strategy_silent_on_thanks():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"]),
        outcome_strategy="follow_up",
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("a fact")
    mem.complete("tell me the fact")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("thanks!")  # politeness is NOT signal
    assert mem.memory.nodes[nid].weight == w_before


def test_complete_goal_retroactively_reinforces_supporting_memories():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["ok"]),
        outcome_strategy="manual",
        intrinsic_retrieval_bump=0.0,
    )
    g = mem.remember("publish paper on cold photosynthesis", metadata={"kind": "goal"})
    a = mem.remember("finding about thylakoid lipids")
    b = mem.remember("finding about scandium triflate catalyst")

    # Simulate two episodes where the goal was active alongside a finding.
    mem.memory.reinforce("research turn 1", [g, a], outcome=0.3)
    mem.memory.reinforce("research turn 2", [g, b], outcome=0.3)

    w_a_before = mem.memory.nodes[a].weight
    w_b_before = mem.memory.nodes[b].weight
    n = mem.memory.complete_goal(g, outcome=+1.0)
    assert n == 2
    assert mem.memory.nodes[a].weight > w_a_before
    assert mem.memory.nodes[b].weight > w_b_before


def test_automemory_manual_strategy_does_not_auto_infer():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["x", "y"]),
        outcome_strategy="manual",
        intrinsic_retrieval_bump=0.0,
    )
    nid = mem.remember("relevant fact")
    mem.complete("query about fact")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("no, that's wrong")  # would-be negative under follow_up
    # manual: no auto-reinforce; intrinsic bump disabled for the test
    assert mem.memory.nodes[nid].weight == w_before


def test_automemory_report_applies_explicit_outcome():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["x"]),
        outcome_strategy="manual",
    )
    nid = mem.remember("relevant fact")
    mem.complete("query about fact")
    w_before = mem.memory.nodes[nid].weight
    assert mem.report(+1.0) is True
    assert mem.memory.nodes[nid].weight > w_before


def test_automemory_report_idempotent_when_no_pending():
    from coherence import AutoMemory

    mem = AutoMemory(chat_fn=_fake_chat_factory(["x"]))
    assert mem.report(+1.0) is False  # nothing to attribute


def test_automemory_custom_outcome_callable():
    from coherence import AutoMemory

    seen: list[tuple[str, str, str]] = []

    def picky(prev_user, prev_reply, next_user):
        seen.append((prev_user, prev_reply, next_user))
        return -0.5

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"]),
        outcome_strategy=picky,
    )
    nid = mem.remember("topic fact")
    mem.complete("first")
    w_before = mem.memory.nodes[nid].weight
    mem.complete("second")
    # Custom strategy applies per turn (no batching); the prior turn should
    # have been graded -0.5 by `picky` and reinforcement applied.
    assert seen and seen[0][2] == "second"
    assert mem.memory.nodes[nid].weight < w_before


def test_automemory_persists_and_reloads(tmp_path):
    from coherence import AutoMemory

    path = tmp_path / "amem.json"
    mem = AutoMemory(chat_fn=_fake_chat_factory(["a"]), path=path)
    mem.remember("first fact")
    mem.complete("anything")
    mem.close()
    assert path.exists()

    mem2 = AutoMemory(chat_fn=_fake_chat_factory(["b"]), path=path)
    assert len(mem2.memory.nodes) >= 1
    # The reloaded memory carries the prior episode count.
    assert mem2.memory.episode_counter >= 0


def test_automemory_reset_history_keeps_memory_and_flushes():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["a", "b"], self_assess_score=+1.0),
        outcome_strategy="self_assess",
        system_prompt="sys",
        judge_batch_size=10,
        intrinsic_retrieval_bump=0.0,
    )
    mem.remember("kept fact")
    mem.complete("first")
    mem.reset_history()
    # History reset to just the system prompt.
    assert len(mem.history) == 1 and mem.history[0]["role"] == "system"
    # Memory survives.
    assert len(mem.memory.nodes) == 1
    # The buffered turn was flushed at the session boundary.
    assert len(mem.memory.experiences) == 1


def test_enrichment_adds_aliases_to_node_metadata():
    from coherence import AutoMemory

    def enriching_chat(messages, tools=None, tool_choice=None, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        if "memory analyzer" in sys:
            return {"choices": [{"message": {"content":
                '{"aliases":["my research focus","what I study"],'
                '"entities":["algae"],"kind":"fact"}'
            }}]}
        return {"choices": [{"message": {"content": "ok"}}]}

    mem = AutoMemory(chat_fn=enriching_chat, outcome_strategy="manual")
    nid = mem.remember("I study extremophile algae.")
    node = mem.memory.nodes[nid]
    assert node.metadata.get("aliases") == ["my research focus", "what I study"]
    assert node.metadata.get("entities") == ["algae"]
    assert node.metadata.get("kind") == "fact"


def test_enrichment_aliases_improve_lexical_recall():
    from coherence import AutoMemory

    def enriching_chat(messages, tools=None, tool_choice=None, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        if "memory analyzer" in sys:
            return {"choices": [{"message": {"content":
                '{"aliases":["my work","what I research"],"entities":[],"kind":"fact"}'
            }}]}
        return {"choices": [{"message": {"content": "ok"}}]}

    mem = AutoMemory(chat_fn=enriching_chat, outcome_strategy="manual")
    mem.remember("I study extremophile algae at the Reykjavik institute.")
    # The original text doesn't contain "my work", but the alias does — so a
    # query using "my work" should still surface the node.
    hits = mem.memory.recall("tell me about my work")
    assert len(hits) == 1
    assert "extremophile algae" in hits[0].text


def test_context_budget_trims_recall_set():
    from coherence import AutoMemory

    fake = _fake_chat_factory(["ok"], self_assess_score=0.0)

    mem = AutoMemory(
        chat_fn=fake,
        outcome_strategy="manual",
        recall_k=10,
        context_budget_tokens=120,   # tight — should clip to 1-2 memories
    )
    for i in range(8):
        # Each text is ~80 chars; budget allows roughly (120*4-400) / ~100 nodes
        mem.remember(f"This is a moderately long fact about subject number {i:03d}.")

    big_set = mem.memory.recall("subject", k=10)
    packed = mem._pack_to_budget(big_set)
    assert len(packed) < len(big_set)
    assert len(packed) >= 1


def test_memory_export_markdown(tmp_path):
    from coherence import Memory

    mem = Memory()
    mem.ingest("alpha fact", tags=["topic-a"])
    mem.ingest("beta fact", tags=["topic-b"], metadata={"kind": "goal"})
    out = tmp_path / "dump.md"
    mem.export(out, format="markdown")
    content = out.read_text()
    assert "# Memory dump" in content
    assert "alpha fact" in content
    assert "beta fact" in content
    assert "## Goals" in content
    assert "## Memories" in content


def test_memory_export_text(tmp_path):
    from coherence import Memory

    mem = Memory()
    mem.ingest("first")
    mem.ingest("second")
    out = tmp_path / "dump.txt"
    mem.export(out, format="text")
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert all("first" in l or "second" in l for l in lines)


def test_memory_export_unknown_format_raises(tmp_path):
    from coherence import Memory

    mem = Memory()
    with pytest.raises(ValueError):
        mem.export(tmp_path / "dump", format="rtf")


def test_memory_goals_lists_goal_kind_nodes():
    from coherence import Memory

    mem = Memory()
    mem.ingest("ordinary fact")
    mem.ingest("publish paper this quarter", metadata={"kind": "goal"})
    goals = mem.goals()
    assert len(goals) == 1
    assert "publish paper" in goals[0].text


# --- LLM recall with size-based fallback -------------------------------------


def test_llm_recall_picks_when_dump_fits():
    from coherence import AutoMemory

    calls: list[str] = []
    base = _fake_chat_factory(["ok"], recall_pick_first=2)

    def tracked(messages, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        calls.append(sys[:60])
        return base(messages, **kw)

    mem = AutoMemory(
        chat_fn=tracked,
        outcome_strategy="manual",
        recall_mode="auto",
        recall_threshold_chars=50_000,
    )
    a = mem.remember("alpha fact about apples")
    b = mem.remember("beta fact about bridges")
    c = mem.remember("gamma fact about galaxies")

    nodes = mem._do_recall("question about apples")
    # The fake recall returns the first two IDs from the listing.
    assert len(nodes) == 2
    assert nodes[0].id in {a, b, c}
    # And the LLM recall path was hit.
    assert any("memory retrieval engine" in c for c in calls)


def test_llm_recall_falls_back_to_bm25_when_dump_too_large():
    from coherence import AutoMemory

    calls: list[str] = []
    base = _fake_chat_factory(["ok"], recall_pick_first=1)

    def tracked(messages, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        calls.append(sys[:60])
        return base(messages, **kw)

    mem = AutoMemory(
        chat_fn=tracked,
        outcome_strategy="manual",
        recall_mode="auto",
        recall_threshold_chars=120,  # very tight
    )
    for txt in [
        "alpha fact about apples and orchards in autumn weather",
        "beta fact about bridges over the river thames at dusk",
        "gamma fact about galaxies and astronomical observation logs",
    ]:
        mem.remember(txt)

    calls.clear()
    nodes = mem._do_recall("apples orchards autumn")
    # Fallback to BM25: the LLM recall path should NOT have been called.
    assert not any("memory retrieval engine" in c for c in calls)
    assert nodes and "alpha" in nodes[0].text


def test_recall_mode_bm25_forces_bm25():
    from coherence import AutoMemory

    calls: list[str] = []
    base = _fake_chat_factory(["ok"], recall_pick_first=2)

    def tracked(messages, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        calls.append(sys[:60])
        return base(messages, **kw)

    mem = AutoMemory(
        chat_fn=tracked,
        outcome_strategy="manual",
        recall_mode="bm25",
        recall_threshold_chars=999_999,  # would normally allow LLM recall
    )
    mem.remember("alpha fact")
    mem.remember("beta fact")

    calls.clear()
    mem._do_recall("alpha")
    assert not any("memory retrieval engine" in c for c in calls)


def test_recall_mode_llm_always_uses_llm():
    from coherence import AutoMemory

    calls: list[str] = []
    base = _fake_chat_factory(["ok"], recall_pick_first=1)

    def tracked(messages, **kw):
        sys = (messages[0].get("content", "") if messages else "") or ""
        calls.append(sys[:60])
        return base(messages, **kw)

    mem = AutoMemory(
        chat_fn=tracked,
        outcome_strategy="manual",
        recall_mode="llm",
        recall_threshold_chars=999_999,
    )
    mem.remember("a fact")
    calls.clear()
    nodes = mem._do_recall("query")
    assert any("memory retrieval engine" in c for c in calls)
    assert len(nodes) == 1


def test_context_length_derives_threshold_and_budget():
    from coherence import AutoMemory

    mem = AutoMemory(
        chat_fn=_fake_chat_factory(["ok"]),
        outcome_strategy="manual",
        context_length=200_000,
    )
    # Threshold should be ~1/4 of context (in chars).
    assert mem.recall_threshold_chars >= 40_000
    # Budget should be ~1/10 of context (in tokens).
    assert mem.context_budget_tokens >= 10_000

    short = AutoMemory(
        chat_fn=_fake_chat_factory(["ok"]),
        outcome_strategy="manual",
        context_length=8_000,
    )
    assert short.recall_threshold_chars < 4_000
    assert short.context_budget_tokens < 1_500


def test_invalid_recall_mode_raises():
    from coherence import AutoMemory

    with pytest.raises(ValueError):
        AutoMemory(chat_fn=_fake_chat_factory(), recall_mode="semantic")
