# Build Plan: An Experiential Memory Framework (Neural-Net-Style Updates, Embedding-Free)

**Audience:** Claude Code (implementation agent).

**One-line goal:** Build a growing, Hermes-style text memory (file-dump nodes the agent reads, edits, and accumulates over sessions) where each node and each connection carries a numeric **weight**, and those weights are updated by a **gradient-descent-style learning rule** driven purely by **task outcomes** — no embeddings, no labeled datasets, no literal autograd. The neural-network analogy is *mimicked in the update mechanism and structure*, not cloned with tensors.

---

## 0. Foundational design commitments (do not violate)

These four decisions define the project. Everything else serves them.

1. **Embedding-free.** Nodes are plain text (Hermes-style markdown dumps). Matching a query to nodes uses lexical/keyword/symbolic overlap (e.g. BM25 / token overlap / tag match), NOT vector embeddings. There is no encoder, no vector store as the substrate. (A keyword index is fine; a learned embedding space is not.)

2. **The only training signal is task outcome.** Each completed agent task yields one scalar: `success` (the answer was correct/accepted) or `failure`. There is no per-node ground truth. There are no labels.

3. **Learning is unsupervised-from-experience, not supervised-from-labels.** The unit of learning is an **Experience** (a.k.a. episode): `{query, active_nodes, outcome, timestamp}`. The system learns about the user and their tasks by living through experiences and adjusting weights — the way Hermes accumulates context, but with numeric reinforcement instead of file rewrites alone. Use the vocabulary **experience / episode / outcome / reinforce / consolidate / decay**, never *label / target / dataset / train-set*.

4. **Neural-net mimicry, not cloning.** The mechanism is a gradient-descent-*style* update rule (the delta rule + Hebbian co-activation — the original embedding-free neural learning rules). Surface API uses memory vocabulary; internal docstrings may note the neural analogy. No PyTorch autograd. No RL library. The "gradient" is a hand-derived, explicitly-coded weight delta.

---

## 1. The learning rule (this is the heart — implement exactly this)

### 1.1 Why this is honestly "gradient-descent-style" without embeddings

The **delta rule** — `Δw = η · (target − output) · input` — is literally the gradient of squared error for a linear unit. It is the original neural-network learning rule, predates embeddings entirely, and operates on scalar signals. We use it. That is what makes "gradient descent on memory, embedding-free" a truthful description rather than marketing.

### 1.2 The objects

- **Node** `n_i`: `{ id, text, weight w_i (scalar, the trainable parameter), activation a_i (transient, per-episode), metadata }`.
- **Edge** `W_ij`: scalar connection weight between nodes `i` and `j` (association strength).
- **Experience** `E`: `{ query, active_nodes (the nodes that were in the agent's context this episode), outcome ∈ {+1 success, −1 failure}, timestamp }`.

### 1.3 Forward pass (retrieval = activation)

Given a query, light up matching nodes and let activation spread one hop over edges (spreading activation — the embedding-free analog of a forward pass):

```
match_i   = lexical_overlap(query, n_i.text)        # BM25 / token / tag overlap, in [0,1]
base_i    = match_i + w_i                            # node weight biases salience
spread_i  = sum_j ( W_ij * base_j )                  # one-hop spread over learned edges
a_i       = squash(base_i + gamma * spread_i)        # squash = sigmoid/tanh -> activation
```

Retrieve the top-k nodes by activation `a_i`. Those become `active_nodes` for the episode. `gamma` controls how much association-spread matters vs direct match.

### 1.4 Backward pass (reinforcement = the weight update)

After the agent answers and we observe `outcome`, update the weights of the nodes that were active. This is the delta rule with an **eligibility trace** to solve credit assignment (see §2):

```
# Per active node i:
elig_i   = a_i / sum_{j active} a_j         # eligibility: share of credit ∝ how active node i was
error    = outcome                          # +1 success, -1 failure (the scalar error signal)
Δw_i     = eta * error * elig_i             # delta-rule-style update
w_i      = w_i + Δw_i

# Hebbian co-activation on edges (nodes that succeed together, wire together):
# for every pair (i,j) both active this episode:
ΔW_ij    = eta_edge * error * elig_i * elig_j
W_ij     = W_ij + ΔW_ij
```

- On **success**, active nodes (weighted by eligibility) get reinforced and their mutual edges strengthen.
- On **failure**, the same nodes get suppressed and their edges weaken — the system learns which memories led it astray.
- `eta` (node learning rate) and `eta_edge` decay over time (a **learning-rate schedule**, mimicked).

### 1.5 Regularization = forgetting (the compression objective)

Two mechanisms, both embedding-free, both standard neural-net regularization mimicked as memory dynamics:

```
# Decay (weight-decay analog): every episode, all weights relax toward zero.
w_i   = w_i * (1 - decay_node)
W_ij  = W_ij * (1 - decay_edge)

# Pruning (L1/sparsity analog): drop nodes/edges whose weight falls below a floor.
if |w_i| < prune_floor:  remove node i (forgotten)
if |W_ij| < edge_floor:  remove edge ij
```

Decay means an unreinforced memory fades; pruning means a faded memory is forgotten. Together they are the "forget what doesn't matter, keep what earns its keep" objective — expressed as weight decay + L1 sparsity, exactly mirroring how a neural net is regularized. This is the compression story, and it requires no embeddings.

### 1.6 What "loss" means here (for honest framing)

There is no differentiable scalar loss being autograd-ed. The implicit objective the update rule hill-climbs is: **maximize task success while minimizing retained memory mass.** The delta-rule update is the ascent step on the success term; the decay+prune is the descent step on the memory-mass term. Document it this way — it is a gradient-descent-*style* optimizer over an implicit objective, not a literal `loss.backward()`.

---

## 2. Credit assignment (the hard problem your constraints create — do not skip)

With only a final success/failure scalar and several active nodes, naive delta-rule rewards every active node equally, including freeloaders. The spec handles this three ways, all already in §1:

1. **Eligibility trace** (`elig_i`): credit is shared in proportion to each node's activation that episode, so the node that actually drove the retrieval gets the most reward/blame. (Standard RL/neuroscience mechanism, used here as plain arithmetic — no RL library.)
2. **Hebbian edges**: co-activation on success builds structure that makes good node-clusters fire together next time, concentrating credit over repeated experiences.
3. **Repetition averages out noise**: a freeloader node that's incidentally active across many unrelated successes gets diffuse small rewards; a genuinely causal node gets concentrated repeated rewards. Over many experiences, causal nodes win. This is why the system needs *volume of experience*, not labels.

Implement an `eligibility` module so this is explicit and tunable, not buried.

---

## 3. Growing memory like Hermes (ingestion + self-editing)

Mirror Hermes' "memory is a living, edited thing," but every dump is a weighted node.

- **Ingest:** new information → new node, `text` = the dump, `w_i` initialized neutral (e.g. 0), keyword-indexed. Node count grows over sessions.
- **Self-edit / consolidate:** periodically, merge near-duplicate nodes (high lexical overlap) into one, summing their weights; rewrite stale node text (Hermes-style "stale memory" cleanup) while preserving the learned weight. Provide a `consolidate()` op.
- **Persist:** save/load the whole memory (node texts + weights + edges) to disk (SQLite or flat files, Hermes-style) so it survives across sessions. This persistence is what makes it *memory*, not a per-session cache.

---

## 4. Module layout

```
expmem/
  __init__.py
  node.py          # Node: text, weight, transient activation, metadata
  graph.py         # Memory: nodes + edge weights; ingest, consolidate, prune, save/load
  matcher.py       # embedding-FREE query->node matching (BM25 / token overlap / tags)
  activation.py    # forward pass: spreading activation -> top-k active nodes
  eligibility.py   # credit-assignment: per-node eligibility from activations
  reinforce.py     # backward pass: delta-rule node update + Hebbian edge update
  forget.py        # decay + pruning (regularization)
  experience.py    # Experience/episode object + the experience loop
  hermes_adapter.py# mine Hermes task loop: episode -> {query, active_nodes, outcome}
  api.py           # public API (memory vocabulary, see below)
tests/
examples/
```

### Public API (sensible mimicry — memory words on top, neural mechanism beneath)

```python
mem = Memory()
mem.ingest(text)                       # add a memory node (grows the store)
active = mem.recall(query, k=5)        # forward pass: spreading activation -> top-k nodes
mem.reinforce(query, active, outcome)  # backward pass: delta-rule + Hebbian from task outcome
mem.consolidate()                      # merge/rewrite nodes (Hermes-style self-edit)
mem.forget()                           # decay + prune (regularization step)
mem.save(path); Memory.load(path)      # persistence across sessions
```

Internally: `recall` runs `activation`, `reinforce` runs `eligibility`+`reinforce`, `forget` runs `forget`. Docstrings may note "delta-rule / Hebbian / weight-decay analog"; surface names stay memory-native.

---

## 5. The experience loop (how it all runs)

```
loop over agent sessions:
    query   = user's request
    active  = mem.recall(query)              # forward pass
    answer  = agent_acts_using(active)       # external; the LLM reads active node texts
    outcome = observe_success(answer)        # +1 / -1, from task result (NOT a label)
    mem.reinforce(query, active, outcome)    # backward pass: weights move
    occasionally: mem.consolidate(); mem.forget()
    mem.ingest(any new durable info from this session)
```

This is online, lifelong, unsupervised learning from lived experience. No training phase, no dataset — it learns continuously about the user and their tasks.

---

## 6. Where outcomes come from (no labels — task signal only)

`outcome` is observed, not labeled. Sources, in priority:

1. **Hermes task loop (primary):** Hermes already tracks task completion and writes skill docs on success. `hermes_adapter` reads that loop: a completed task → `{query, the nodes that were in context (active_nodes), success/failure}`. This is the natural outcome stream.
2. **Explicit user feedback (secondary):** thumbs-up/down, acceptance, correction = outcome signal about the user and their tasks.
3. **Implicit signals (optional):** user reused the answer / didn't re-ask = success; user rephrased or corrected = failure.

None of these are labels in the supervised sense — they're observed consequences of the agent acting, used as the scalar error in §1.4.

---

## 7. Phased build

**Phase 1 — Static forward pass.** `node`, `matcher` (embedding-free), `activation` (no edges yet, no spread), `recall()`. Verify top-k recall by lexical match works. No learning yet.

**Phase 2 — Reinforcement (the core).** Add `eligibility`, `reinforce` (delta-rule node updates from outcome), and the experience loop. **Gate:** over a sequence of simulated experiences where some nodes are genuinely causal for success, those nodes' weights rise above the freeloaders'. If causal nodes don't separate from incidental ones, the credit-assignment isn't working — fix before proceeding.

**Phase 3 — Edges + spreading activation.** Add `W_ij`, Hebbian co-activation updates, one-hop spread in the forward pass. **Gate:** co-activated successful nodes form edges that improve recall of related nodes the lexical matcher alone would miss.

**Phase 4 — Forgetting.** Add `forget` (decay + prune). **Gate:** unreinforced nodes fade and get pruned while reinforced ones persist; total node count stabilizes instead of growing unbounded. Produce the curve: task success vs. nodes retained.

**Phase 5 — Hermes integration + persistence.** `hermes_adapter`, `consolidate`, `save/load`. Run the live experience loop against real Hermes sessions.

**Phase 6 — Package & ship** as an OSS embedding-free experiential memory layer / Hermes provider.

---

## 8. Evaluation

Report jointly (never one alone):
1. **Task success rate** over time — does it climb as the system accumulates experience?
2. **Nodes retained** after forgetting — is memory staying compressed?
3. **Latency** of `recall` (spreading activation over the graph).

Compare against: a no-learning lexical-recall baseline (does reinforcement actually help?), and a Hermes-flat-file baseline (does the weighted graph beat plain file memory?). The headline claim to validate: **success rate rises and memory size stays bounded as experience accumulates — without any embeddings or labels.**

---

## 9. Honest-framing requirements (bake into README)

- This is **gradient-descent-*style***: the delta rule is the gradient of squared error for a linear unit, applied to memory weights — a faithful mimic, not literal autograd. Say so plainly.
- It is **embedding-free**: matching is lexical/symbolic; weights are scalars on text nodes.
- It learns **from experience, not labels**: the only signal is observed task outcome.
- **Prior art to cite, not hide:** Hebbian learning (the co-activation rule), the delta/perceptron rule (the node update), spreading-activation theory and ACT-R (the forward pass), eligibility traces (credit assignment), and the recent RL-trained memory-graph papers (arXiv 2511.07800, HAGE) as the *reward-driven* cousins of this *outcome-driven* approach. 
- **Defensible novelty:** an embedding-free, label-free, persistent text-memory graph (Hermes-style) whose node and edge weights self-organize by a delta-rule + Hebbian update driven solely by task outcome, with decay+pruning as a built-in compression mechanism. The combination — neural-net-style learning dynamics on plain editable text memory, no embeddings, no labels — is the contribution. Do not claim literal backprop or "first trainable memory."

---

## 10. First concrete tasks for the implementation agent

1. Scaffold the package in §4.
2. Build `matcher.py` (BM25 or token-overlap) and `node.py`.
3. Build `activation.py` + `recall()` (Phase 1), verify lexical top-k.
4. Build `eligibility.py` + `reinforce.py` + the experience loop (Phase 2).
5. Write a simulation: a set of nodes where a known subset is causal for success; run many experiences; show the causal nodes' weights separate from the rest. Report that separation before building Phase 3.
