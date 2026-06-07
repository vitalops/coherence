from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .node import Node, new_id
from .matcher import LexicalIndex, tokenize, token_jaccard
from .activation import forward_pass
from .eligibility import eligibility as _eligibility
from .reinforce import reinforce_nodes, reinforce_edges
from .forget import decay as _decay, prune as _prune
from .experience import Experience


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


class Memory:
    def __init__(
        self,
        *,
        eta: float = 0.15,
        eta_edge: float = 0.05,
        decay_node: float = 0.01,
        decay_edge: float = 0.02,
        prune_floor: float = 0.02,
        edge_floor: float = 0.01,
        gamma: float = 0.5,
        initial_weight: float = 0.1,
        k_default: int = 5,
        squash: str = "tanh",
        eligibility_kind: str = "proportional",
        eligibility_temperature: float = 0.5,
        min_age_seconds: float = 0.0,
        experience_log_cap: int = 5000,
    ) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str], float] = {}
        self.index = LexicalIndex()
        self.experiences: list[Experience] = []
        self.ingest_enricher: Callable[[str], dict[str, Any]] | None = None

        self.eta = eta
        self.eta_edge = eta_edge
        self.decay_node = decay_node
        self.decay_edge = decay_edge
        self.prune_floor = prune_floor
        self.edge_floor = edge_floor
        self.gamma = gamma
        self.initial_weight = initial_weight
        self.k_default = k_default
        self.squash = squash
        self.eligibility_kind = eligibility_kind
        self.eligibility_temperature = eligibility_temperature
        self.min_age_seconds = min_age_seconds
        self.experience_log_cap = experience_log_cap

        self.episode_counter = 0

    # ------------------------------------------------------------------ ingest

    def ingest(
        self,
        text: str,
        *,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        initial_weight: float | None = None,
        node_id: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("ingest(text=...): text must be non-empty")
        nid = node_id or new_id()
        text = text.strip()

        final_metadata = dict(metadata or {})
        final_aliases = list(aliases or [])

        # If no explicit aliases were passed and an LLM enricher is wired up,
        # ask it once for aliases / entities / kind. Failures are silent and
        # the node still gets ingested (with empty aliases).
        if not aliases and self.ingest_enricher is not None:
            try:
                enrichment = self.ingest_enricher(text) or {}
            except Exception:
                enrichment = {}
            if enrichment:
                aliases_from_llm = enrichment.get("aliases") or []
                final_aliases.extend(a for a in aliases_from_llm if a)
                if aliases_from_llm:
                    final_metadata.setdefault("aliases", list(aliases_from_llm))
                if enrichment.get("entities"):
                    final_metadata.setdefault("entities", list(enrichment["entities"]))
                if enrichment.get("kind"):
                    final_metadata.setdefault("kind", str(enrichment["kind"]))

        node = Node(
            id=nid,
            text=text,
            weight=float(self.initial_weight if initial_weight is None else initial_weight),
            tags=list(tags or []),
            metadata=final_metadata,
        )
        self.nodes[nid] = node
        self.index.add(nid, node.text, aliases=final_aliases or None)
        return nid

    def ingest_many(self, texts: Iterable[str]) -> list[str]:
        return [self.ingest(t) for t in texts]

    def edit(self, node_id: str, new_text: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        node = self.nodes[node_id]
        node.text = new_text.strip()
        aliases = node.metadata.get("aliases") or []
        self.index.update(node_id, node.text, aliases=list(aliases) if aliases else None)

    def goals(self) -> list[Node]:
        return [n for n in self.nodes.values() if str(n.metadata.get("kind") or "").lower() == "goal"]

    def complete_goal(
        self,
        goal_id: str,
        outcome: float = 1.0,
        *,
        decay_per_episode: float = 0.9,
        max_episodes: int = 50,
    ) -> int:
        """Retroactively reinforce memories that supported a goal.

        When a goal is achieved (or definitively missed), this method walks
        the experience log for episodes whose active set included the goal
        node, and applies a reinforcement to each of those active sets with
        the given outcome, weighted by recency (the most recent episode
        gets full credit, earlier ones get geometrically less).

        Returns the number of episodes reinforced.

        Parameters
        ----------
        goal_id : str
            The id of the goal node whose pursuit is being graded.
        outcome : float
            +1 if the goal was achieved (boosts supporting memories),
            -1 if it was missed (demotes them). Graded values are fine.
        decay_per_episode : float
            Per-step recency weighting; episodes older than the current one
            get progressively less credit.
        max_episodes : int
            Cap on how many matching episodes to walk back through.
        """
        if goal_id not in self.nodes:
            return 0
        # Reinforce the goal node itself directly (a strong, undiluted signal).
        self.reinforce(
            query=f"goal completion: {self.nodes[goal_id].text[:80]}",
            active=[goal_id],
            outcome=float(outcome),
            metadata={"inferred_by": "complete_goal", "goal_id": goal_id},
        )
        # Walk the episode log from the most recent and reinforce active sets
        # that included this goal, with geometric recency decay.
        reinforced = 0
        weight = 1.0
        for exp in reversed(self.experiences):
            if reinforced >= max_episodes:
                break
            if goal_id not in exp.active_ids:
                continue
            other_active = [aid for aid in exp.active_ids if aid != goal_id and aid in self.nodes]
            if not other_active:
                continue
            self.reinforce(
                query=exp.query,
                active=other_active,
                outcome=float(outcome) * weight,
                metadata={"inferred_by": "complete_goal", "goal_id": goal_id, "from_episode": exp.episode},
            )
            reinforced += 1
            weight *= decay_per_episode
        return reinforced

    def remove(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.index.remove(node_id)
        for key in list(self.edges.keys()):
            if node_id in key:
                del self.edges[key]
        return True

    # ---------------------------------------------------------------- forward

    def recall(self, query: str, k: int | None = None) -> list[Node]:
        if not self.nodes:
            return []
        k = k if k is not None else self.k_default
        _activations, active_ids = forward_pass(
            query,
            self.nodes,
            self.edges,
            self.index,
            k=k,
            gamma=self.gamma,
            squash=self.squash,
        )
        return [self.nodes[nid] for nid in active_ids]

    def activations(self, query: str) -> dict[str, float]:
        if not self.nodes:
            return {}
        a, _ = forward_pass(
            query,
            self.nodes,
            self.edges,
            self.index,
            k=len(self.nodes),
            gamma=self.gamma,
            squash=self.squash,
        )
        return a

    # ----------------------------------------------------------- backward / log

    def reinforce(
        self,
        query: str,
        active: Iterable[Node | str],
        outcome: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Experience | None:
        active_ids = []
        for item in active:
            nid = item if isinstance(item, str) else item.id
            if nid in self.nodes and nid not in active_ids:
                active_ids.append(nid)
        if not active_ids:
            return None

        # Recompute against current graph to avoid stale per-node activations.
        activations_map, _ = forward_pass(
            query,
            self.nodes,
            self.edges,
            self.index,
            k=len(active_ids),
            gamma=self.gamma,
            squash=self.squash,
        )

        elig = _eligibility(
            activations_map,
            active_ids,
            kind=self.eligibility_kind,
            temperature=self.eligibility_temperature,
        )
        reinforce_nodes(self.nodes, active_ids, outcome, elig, self.eta)
        reinforce_edges(self.edges, active_ids, outcome, elig, self.eta_edge)

        exp = Experience(
            query=query,
            active_ids=active_ids,
            outcome=float(outcome),
            episode=self.episode_counter,
            metadata=dict(metadata or {}),
        )
        self.experiences.append(exp)
        if len(self.experiences) > self.experience_log_cap:
            self.experiences = self.experiences[-self.experience_log_cap:]
        self.episode_counter += 1
        return exp

    # ------------------------------------------------------------ regularize

    def forget(self) -> dict[str, list]:
        _decay(self.nodes, self.edges, self.decay_node, self.decay_edge)
        removed_nodes, removed_edges = _prune(
            self.nodes,
            self.edges,
            self.index,
            self.prune_floor,
            self.edge_floor,
            min_age_seconds=self.min_age_seconds,
        )
        return {"removed_nodes": removed_nodes, "removed_edges": removed_edges}

    def consolidate(
        self,
        *,
        similarity_threshold: float = 0.82,
        max_pairs: int = 1024,
    ) -> list[tuple[str, str]]:
        merges: list[tuple[str, str]] = []
        ids = list(self.nodes.keys())
        absorbed: set[str] = set()
        for i, ni in enumerate(ids):
            if ni in absorbed:
                continue
            if len(merges) >= max_pairs:
                break
            for nj in ids[i + 1:]:
                if nj in absorbed:
                    continue
                if (
                    token_jaccard(self.nodes[ni].text, self.nodes[nj].text)
                    >= similarity_threshold
                ):
                    self._merge_into(ni, nj)
                    absorbed.add(nj)
                    merges.append((ni, nj))
                    if len(merges) >= max_pairs:
                        break
        return merges

    def _merge_into(self, keep_id: str, drop_id: str) -> None:
        keep = self.nodes[keep_id]
        drop = self.nodes[drop_id]
        keep.weight += drop.weight
        keep.reinforcement_count += drop.reinforcement_count
        keep.failure_count += drop.failure_count
        # Union of tags / metadata; preserve unique values
        keep.tags = list(dict.fromkeys(list(keep.tags) + list(drop.tags)))
        for k, v in drop.metadata.items():
            keep.metadata.setdefault(k, v)
        for key in list(self.edges.keys()):
            if drop_id not in key:
                continue
            other = key[0] if key[1] == drop_id else key[1]
            w = self.edges.pop(key)
            if other == keep_id:
                continue  # internal edge of the merged pair
            new_key = _edge_key(keep_id, other)
            self.edges[new_key] = self.edges.get(new_key, 0.0) + w
        del self.nodes[drop_id]
        self.index.remove(drop_id)

    # ------------------------------------------------------------ inspection

    def neighbors(self, node_id: str) -> list[tuple[str, float]]:
        out = []
        for (i, j), w in self.edges.items():
            if i == node_id:
                out.append((j, w))
            elif j == node_id:
                out.append((i, w))
        out.sort(key=lambda kv: -abs(kv[1]))
        return out

    def top_nodes(self, n: int = 10) -> list[Node]:
        return sorted(self.nodes.values(), key=lambda nd: -nd.weight)[:n]

    def stats(self) -> dict[str, Any]:
        weights = [n.weight for n in self.nodes.values()]
        edge_w = list(self.edges.values())
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "episodes": self.episode_counter,
            "experiences_retained": len(self.experiences),
            "mean_node_weight": sum(weights) / max(len(weights), 1),
            "max_node_weight": max(weights) if weights else 0.0,
            "min_node_weight": min(weights) if weights else 0.0,
            "mean_edge_weight": sum(edge_w) / max(len(edge_w), 1),
            "max_edge_weight": max(edge_w) if edge_w else 0.0,
        }

    def recall_as_context(
        self,
        query: str,
        k: int | None = None,
        *,
        header: str = "## Relevant memories",
        include_weight: bool = False,
    ) -> str:
        nodes = self.recall(query, k=k)
        if not nodes:
            return ""
        lines = [header]
        for n in nodes:
            tag = f"[{n.id[:8]}"
            if include_weight:
                tag += f" w={n.weight:.3f}"
            tag += "]"
            lines.append(f"- {tag} {n.text}")
        return "\n".join(lines)

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "config": {
                "eta": self.eta,
                "eta_edge": self.eta_edge,
                "decay_node": self.decay_node,
                "decay_edge": self.decay_edge,
                "prune_floor": self.prune_floor,
                "edge_floor": self.edge_floor,
                "gamma": self.gamma,
                "initial_weight": self.initial_weight,
                "k_default": self.k_default,
                "squash": self.squash,
                "eligibility_kind": self.eligibility_kind,
                "eligibility_temperature": self.eligibility_temperature,
                "min_age_seconds": self.min_age_seconds,
                "experience_log_cap": self.experience_log_cap,
            },
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {"i": k[0], "j": k[1], "w": float(w)} for k, w in self.edges.items()
            ],
            "experiences": [e.to_dict() for e in self.experiences],
            "episode_counter": self.episode_counter,
            "saved_at": time.time(),
        }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        p = Path(path)
        data = json.loads(p.read_text())
        cfg = data.get("config", {})
        mem = cls(**cfg)
        for ndata in data.get("nodes", []):
            node = Node.from_dict(ndata)
            mem.nodes[node.id] = node
            aliases = node.metadata.get("aliases") or []
            mem.index.add(node.id, node.text, aliases=list(aliases) if aliases else None)
        for edata in data.get("edges", []):
            mem.edges[_edge_key(edata["i"], edata["j"])] = float(edata["w"])
        for exp in data.get("experiences", []):
            mem.experiences.append(Experience.from_dict(exp))
        mem.episode_counter = int(data.get("episode_counter", 0))
        return mem

    def export(self, path: str | Path, format: str = "markdown") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()
        if fmt in ("markdown", "md"):
            p.write_text(self._to_markdown())
        elif fmt == "json":
            self.save(p)
        elif fmt in ("text", "txt", "plain"):
            p.write_text(self._to_text())
        else:
            raise ValueError(f"Unknown export format: {format!r}. Use 'markdown', 'json', or 'text'.")

    def _to_markdown(self) -> str:
        stats = self.stats()
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        lines: list[str] = [
            "# Memory dump",
            "",
            f"_Generated at {ts}_",
            "",
            f"- **Nodes**: {stats['nodes']}",
            f"- **Associations**: {stats['edges']}",
            f"- **Episodes**: {stats['episodes']}",
            f"- **Mean node weight**: {stats['mean_node_weight']:+.3f}",
            "",
        ]

        # Goals first, since they shape the rest.
        goals = self.goals()
        if goals:
            lines.append("## Goals")
            lines.append("")
            for n in sorted(goals, key=lambda x: -x.weight):
                lines.append(f"- **`{n.id[:8]}`** (w={n.weight:+.3f}): {n.text}")
            lines.append("")

        # Group remaining nodes by tag (or kind from enrichment, or "untagged").
        bucket: dict[str, list[Node]] = {}
        for n in sorted(self.nodes.values(), key=lambda x: -x.weight):
            if n in goals:
                continue
            keys = list(n.tags) if n.tags else []
            kind = str(n.metadata.get("kind") or "").strip()
            if kind and kind != "goal" and kind not in keys:
                keys.append(kind)
            if not keys:
                keys = ["(untagged)"]
            for k in keys:
                bucket.setdefault(k, []).append(n)

        lines.append("## Memories")
        lines.append("")
        for tag in sorted(bucket.keys()):
            lines.append(f"### {tag}")
            lines.append("")
            for n in bucket[tag]:
                entities = n.metadata.get("entities") or []
                ent_str = f" — entities: {', '.join(entities)}" if entities else ""
                lines.append(
                    f"- **`{n.id[:8]}`** (w={n.weight:+.3f}, "
                    f"reinforced {n.reinforcement_count}, failed {n.failure_count}): "
                    f"{n.text}{ent_str}"
                )
            lines.append("")

        edges_sorted = sorted(self.edges.items(), key=lambda kv: -abs(kv[1]))[:50]
        if edges_sorted:
            lines.append("## Strongest associations")
            lines.append("")
            for (i, j), w in edges_sorted:
                if i in self.nodes and j in self.nodes:
                    ti = self.nodes[i].text[:80].replace("\n", " ")
                    tj = self.nodes[j].text[:80].replace("\n", " ")
                    lines.append(f"- `{w:+.3f}`  `{i[:8]}` ↔ `{j[:8]}`  —  \"{ti}\" / \"{tj}\"")
            lines.append("")

        if self.experiences:
            lines.append("## Recent experiences (last 20)")
            lines.append("")
            for e in self.experiences[-20:]:
                q = e.query[:90].replace("\n", " ")
                lines.append(f"- episode {e.episode}, outcome {e.outcome:+.2f}: \"{q}\"")
            lines.append("")

        return "\n".join(lines)

    def _to_text(self) -> str:
        out: list[str] = []
        for n in sorted(self.nodes.values(), key=lambda x: -x.weight):
            out.append(f"[{n.weight:+.3f}] {n.text}")
        return "\n".join(out)


__all__ = ["Memory"]
