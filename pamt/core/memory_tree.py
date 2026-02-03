from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

from ..config import UpdateConfig
from ..embeddings.models import EmbeddingClient
from ..llms.models import LLM
from .types import PreferenceFusion, PreferenceVector, CategoryWithProb
from .update import PAMTState, PAMTUpdater

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    name: str
    state: PAMTState = field(default_factory=PAMTState)
    children: Dict[str, "MemoryNode"] = field(default_factory=dict)
    embedding: List[float] | None = None
    label: str | None = None
    content_embedding: List[float] | None = None
    content_count: int = 0
    recent_texts: List[str] = field(default_factory=list)

    def get_child(self, name: str) -> "MemoryNode | None":
        return self.children.get(name)

    def get_or_create_child(self, name: str) -> "MemoryNode":
        node = self.children.get(name)
        if node is None:
            node = MemoryNode(name=name, label=name)
            self.children[name] = node
        return node


@dataclass
class RetrievalConfig:
    similarity_strict: float = 0.75
    similarity_loose: float = 0.6
    max_candidates: int = 3
    max_leaves_per_category: int = 8
    leaf_reuse_threshold: float = 0.75
    leaf_merge_threshold: float = 0.6
    leaf_summary_max_samples: int = 4
    short_query_max_chars: int = 6


@dataclass
class HierarchicalMemoryTree:
    """3-layer memory tree with weighted bottom-up updates and fallback retrieval."""
    update_config: UpdateConfig
    retrieval_config: RetrievalConfig
    embedder: EmbeddingClient
    llm: LLM
    label_prompt_path: str = "prompts/category_leaf.txt"
    leaf_prompt_path: str = "prompts/leaf_only.txt"
    merge_prompt_path: str = "prompts/leaf_merge_summary.txt"
    root_name: str = "root"
    updater: PAMTUpdater = field(init=False)
    root: MemoryNode = field(init=False)
    _label_prompt_cache: str | None = field(default=None, init=False, repr=False)
    _leaf_prompt_cache: str | None = field(default=None, init=False, repr=False)
    _merge_prompt_cache: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.retrieval_config.similarity_strict < self.retrieval_config.similarity_loose:
            raise ValueError("similarity_strict must be >= similarity_loose.")
        self.updater = PAMTUpdater(self.update_config)
        self.root = MemoryNode(name=self.root_name, label=self.root_name)

    def get_preference(self, query: str) -> PreferenceFusion | None:
        """Retrieve preferences by embedding-based routing from the root."""
        query = query.strip()
        if not query:
            return self.updater.current_fusion(self.root.state)
        query_embedding = self.embedder.embed(query)
        return self._retrieve_from_node(self.root, query, query_embedding)

    def get_preference_trace(self, query: str) -> Tuple[PreferenceFusion | None, Dict[str, Any]]:
        """Retrieve preferences and return a trace of routing decisions."""
        trace: Dict[str, Any] = {
            "query": query,
            "steps": [],
            "thresholds": {
                "strict": self.retrieval_config.similarity_strict,
                "loose": self.retrieval_config.similarity_loose,
            },
        }
        query = query.strip()
        if not query:
            trace["strategy"] = "empty_query"
            trace["selected_path"] = [self.root.name]
            return self.updater.current_fusion(self.root.state), trace

        embed_start = perf_counter()
        query_embedding = self.embedder.embed(query)
        trace["embedding_ms"] = (perf_counter() - embed_start) * 1000

        pref, selected_path, strategy = self._retrieve_from_node_trace(
            self.root,
            query,
            query_embedding,
            [self.root.name],
            trace["steps"],
        )
        trace["strategy"] = strategy
        trace["selected_path"] = selected_path
        return pref, trace

    def update_preference(
        self,
        path: List[str],
        pref: PreferenceVector,
        content_text: str | None = None,
    ) -> None:
        """Update leaf preference and propagate upwards with attenuated weights."""
        if len(path) != 2:
            raise ValueError("update_preference expects [category, leaf] path.")
        category_name, leaf_name = path

        category = self.root.get_or_create_child(category_name)
        leaf = category.get_or_create_child(leaf_name)
        if leaf.label is None:
            leaf.label = leaf.name
        self._update_leaf_content(leaf, content_text)

        # Leaf gets the full update.
        self.updater.update(leaf.state, pref, weight=1.0)

        # Category weight scales by the proportion of the updated leaf within the category.
        category_leaf_count = max(len(category.children), 1)
        category_weight = 1.0 / category_leaf_count
        self.updater.update(category.state, pref, weight=category_weight)

        # Root weight scales by the category's leaf count within the entire tree.
        total_leaf_count = sum(len(node.children) for node in self.root.children.values())
        root_weight = 0.0
        if total_leaf_count > 0:
            root_weight = 1.0 / total_leaf_count
            self.updater.update(self.root.state, pref, weight=root_weight)
        logger.debug(
            "MemoryTree.update_preference: path=%s leaf_w=1.0 category_w=%.3f root_w=%.3f",
            path,
            category_weight,
            root_weight,
        )

    def update_preference_trace(
        self,
        path: List[str],
        pref: PreferenceVector,
        content_text: str | None = None,
    ) -> Dict[str, Any]:
        """Update preferences and return the weights used for each node."""
        if len(path) != 2:
            raise ValueError("update_preference expects [category, leaf] path.")
        category_name, leaf_name = path

        category = self.root.get_or_create_child(category_name)
        leaf = category.get_or_create_child(leaf_name)
        if leaf.label is None:
            leaf.label = leaf.name
        self._update_leaf_content(leaf, content_text)

        leaf_before = self._fusion_payload(self.updater.current_fusion(leaf.state))
        category_before = self._fusion_payload(self.updater.current_fusion(category.state))
        root_before = self._fusion_payload(self.updater.current_fusion(self.root.state))

        category_leaf_count = max(len(category.children), 1)
        category_weight = 1.0 / category_leaf_count
        total_leaf_count = sum(len(node.children) for node in self.root.children.values())
        root_weight = (1.0 / total_leaf_count) if total_leaf_count > 0 else 0.0

        leaf_fusion, leaf_change = self.updater.update(leaf.state, pref, weight=1.0)
        category_fusion, category_change = self.updater.update(category.state, pref, weight=category_weight)
        root_fusion = None
        root_change = None
        if total_leaf_count > 0:
            root_fusion, root_change = self.updater.update(self.root.state, pref, weight=root_weight)

        return {
            "path": [category_name, leaf_name],
            "nodes": {"root": self.root.name, "category": category_name, "leaf": leaf_name},
            "weights": {"leaf": 1.0, "category": category_weight, "root": root_weight},
            "category_leaf_count": category_leaf_count,
            "total_leaf_count": total_leaf_count,
            "changes": {
                "leaf": {
                    "before": leaf_before,
                    "after": self._fusion_payload(leaf_fusion),
                    "change_signal": self._change_payload(leaf_change),
                },
                "category": {
                    "before": category_before,
                    "after": self._fusion_payload(category_fusion),
                    "change_signal": self._change_payload(category_change),
                },
                "root": {
                    "before": root_before,
                    "after": self._fusion_payload(root_fusion)
                    if root_fusion is not None
                    else root_before,
                    "change_signal": self._change_payload(root_change),
                },
            },
        }

    def route_response(self, query: str, response_text: str | None = None) -> List[str]:
        """Infer a [category, leaf] path from the user query + assistant response."""
        if response_text is None:
            response_text = ""
        combined_text = self._combine_text(query, response_text)
        reused_path = self._reuse_existing_leaf(combined_text)
        if reused_path is not None:
            return reused_path
        category_label, leaf_label = self._infer_labels(query, response_text)
        category = self.root.get_or_create_child(category_label)
        if category.label is None:
            category.label = category.name
        leaf = self._select_or_create_leaf(
            category,
            combined_text,
            leaf_label,
            query,
            response_text,
        )
        return [category.name, leaf.name]

    def route_query(self, query: str, response_text: str | None = None) -> List[str]:
        """Legacy alias: route with query + optional response (prefer route_response)."""
        return self.route_response(query, response_text)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the tree and node preferences."""
        return self._node_snapshot(self.root, [self.root.name])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory tree (structure + state) to a dict."""
        return {
            "root_name": self.root_name,
            "root": self._node_to_dict(self.root),
        }

    def load_from_dict(self, payload: Dict[str, Any]) -> None:
        """Load tree structure + state from a dict."""
        root_payload = payload.get("root", payload)
        root = self._node_from_dict(root_payload)
        if root.name != self.root_name:
            root.name = self.root_name
            if root.label is None:
                root.label = self.root_name
        self.root = root

    def save(self, path: str | Path) -> None:
        """Persist the tree to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), ensure_ascii=True, indent=2)
        target.write_text(data, encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load tree state from a JSON file."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Memory tree file not found: {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Memory tree file must contain a JSON object.")
        self.load_from_dict(data)

    def _retrieve_from_node(
        self,
        node: MemoryNode,
        query: str,
        query_embedding: List[float],
    ) -> PreferenceFusion | None:
        if not node.children:
            return self.updater.current_fusion(node.state)

        scored_children = self._score_children(node, query_embedding)
        strong, weak = self._partition_candidates(scored_children)

        if strong:
            best_child = strong[0][1]
            return self._retrieve_from_node(best_child, query, query_embedding)

        if weak:
            merged = self._merge_candidate_preferences(weak, query, query_embedding)
            if merged is not None:
                return merged

        return self.updater.current_fusion(node.state)

    def _score_children(
        self,
        node: MemoryNode,
        query_embedding: List[float],
    ) -> List[Tuple[float, MemoryNode]]:
        scored: List[Tuple[float, MemoryNode]] = []
        for child in node.children.values():
            child_embedding = self._get_child_embedding(child)
            score = self._cosine_similarity(query_embedding, child_embedding)
            scored.append((score, child))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _partition_candidates(
        self,
        scored_children: List[Tuple[float, MemoryNode]],
    ) -> Tuple[List[Tuple[float, MemoryNode]], List[Tuple[float, MemoryNode]]]:
        strict = self.retrieval_config.similarity_strict
        loose = self.retrieval_config.similarity_loose
        strong = [(score, node) for score, node in scored_children if score >= strict]
        weak = [(score, node) for score, node in scored_children if loose <= score < strict]
        return strong, weak[: self.retrieval_config.max_candidates]

    def _retrieve_from_node_trace(
        self,
        node: MemoryNode,
        query: str,
        query_embedding: List[float],
        path: List[str],
        steps: List[Dict[str, Any]],
    ) -> Tuple[PreferenceFusion | None, List[str], str]:
        step: Dict[str, Any] = {"node_path": path, "children": []}
        if not node.children:
            if node is self.root:
                step["decision"] = "fallback_root_empty"
            else:
                step["decision"] = "fallback_leaf_empty"
            steps.append(step)
            return self.updater.current_fusion(node.state), path, "fallback"

        scored_children = self._score_children(node, query_embedding)
        strict = self.retrieval_config.similarity_strict
        loose = self.retrieval_config.similarity_loose
        for score, child in scored_children:
            if score >= strict:
                bucket = "strong"
            elif score >= loose:
                bucket = "weak"
            else:
                bucket = "low"
            step["children"].append({"name": child.name, "score": score, "bucket": bucket})

        strong, weak = self._partition_candidates(scored_children)

        if strong:
            best_child = strong[0][1]
            step["decision"] = "strong_match"
            step["selected_child"] = best_child.name
            steps.append(step)
            return self._retrieve_from_node_trace(
                best_child,
                query,
                query_embedding,
                path + [best_child.name],
                steps,
            )

        if weak:
            step["decision"] = "weak_merge"
            step["merged_children"] = [
                {"name": child.name, "score": score} for score, child in weak
            ]
            merged = self._merge_candidate_preferences(weak, query, query_embedding)
            if merged is not None:
                steps.append(step)
                return merged, path, "weak_merge"

        if node is self.root:
            step["decision"] = "fallback_root_no_match"
        else:
            step["decision"] = "fallback_no_match"
        steps.append(step)
        return self.updater.current_fusion(node.state), path, "fallback"

    def _merge_candidate_preferences(
        self,
        candidates: List[Tuple[float, MemoryNode]],
        query: str,
        query_embedding: List[float],
    ) -> PreferenceFusion | None:
        prefs: List[PreferenceFusion] = []
        weights: List[float] = []
        for score, node in candidates:
            pref = self._retrieve_from_node(node, query, query_embedding)
            if pref is None:
                continue
            prefs.append(pref)
            weights.append(score)
        if not prefs:
            return None
        return self._weighted_merge(prefs, weights)

    def _infer_labels(self, query: str, response_text: str) -> Tuple[str, str]:
        prompt = self._fill_prompt(self._load_label_prompt(), query=query, response=response_text)
        llm_response = self.llm.generate(prompt)
        data = self._extract_json(llm_response)
        category = self._sanitize_label(data.get("category", "General"), fallback="General")
        leaf = self._sanitize_label(data.get("leaf", "Task"), fallback="Task")
        leaf = self._coarsen_leaf_label(leaf)
        return category, leaf

    def _infer_leaf_label(self, query: str, response_text: str, category_name: str) -> str:
        prompt = self._fill_prompt(
            self._load_leaf_prompt(),
            query=query,
            response=response_text,
            category=category_name,
        )
        llm_response = self.llm.generate(prompt)
        data = self._extract_json(llm_response)
        leaf = self._sanitize_label(data.get("leaf", "Task"), fallback="Task")
        return self._coarsen_leaf_label(leaf)

    def _summarize_leaf_label(
        self,
        category: str,
        leaf_label: str,
        samples: List[str],
        query: str,
        response_text: str,
    ) -> str | None:
        prompt = self._fill_prompt(
            self._load_merge_prompt(),
            query=query,
            response=response_text,
            category=category,
            leaf=leaf_label,
            samples=self._format_samples(samples),
        )
        try:
            llm_response = self.llm.generate(prompt)
        except Exception:
            return None
        data = self._extract_json(llm_response)
        if not data:
            return None
        summary = self._sanitize_label(data.get("leaf", ""), fallback="")
        if not summary:
            return None
        return self._coarsen_leaf_label(summary)

    def _load_label_prompt(self) -> str:
        return self._load_prompt(self.label_prompt_path, "_label_prompt_cache")

    def _load_leaf_prompt(self) -> str:
        return self._load_prompt(self.leaf_prompt_path, "_leaf_prompt_cache")

    def _load_merge_prompt(self) -> str:
        return self._load_prompt(self.merge_prompt_path, "_merge_prompt_cache")

    def _load_prompt(self, prompt_path: str, cache_attr: str) -> str:
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached
        path = Path(prompt_path)
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / prompt_path
        text = path.read_text(encoding="utf-8")
        setattr(self, cache_attr, text)
        return text

    @staticmethod
    def _fill_prompt(
        prompt: str,
        *,
        query: str,
        response: str,
        category: str | None = None,
        leaf: str | None = None,
        samples: str | None = None,
    ) -> str:
        filled = prompt.replace("{{query}}", query).replace("{{response}}", response)
        if category is not None:
            filled = filled.replace("{{category}}", category)
        if leaf is not None:
            filled = filled.replace("{{leaf}}", leaf)
        if samples is not None:
            filled = filled.replace("{{samples}}", samples)
        return filled

    @staticmethod
    def _coarsen_leaf_label(label: str) -> str:
        # If multiple directions appear, keep the first segment.
        for sep in (" / ", "/", ";", ",", "|", " & ", " and "):
            if sep in label:
                label = label.split(sep)[0].strip()
                break
        words = label.split()
        if len(words) > 4:
            label = " ".join(words[:4])
        return label

    @staticmethod
    def _extract_json(text: str) -> Dict[str, str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        payload = text[start : end + 1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _sanitize_label(label: str, fallback: str) -> str:
        cleaned = " ".join(label.strip().strip("\"'").split())
        if not cleaned:
            return fallback
        return cleaned[:64]

    @staticmethod
    def _combine_text(query: str, response_text: str) -> str:
        query = query.strip()
        response_text = response_text.strip()
        if query and response_text:
            return f"User: {query}\nAssistant: {response_text}"
        if query:
            return f"User: {query}"
        if response_text:
            return f"Assistant: {response_text}"
        return ""

    def _format_samples(self, samples: List[str]) -> str:
        if not samples:
            return "None"
        max_samples = self.retrieval_config.leaf_summary_max_samples
        if max_samples > 0 and len(samples) > max_samples:
            samples = samples[-max_samples:]
        lines = []
        for idx, text in enumerate(samples, start=1):
            lines.append(f"{idx}. {text}")
        return "\n".join(lines)

    def _get_embedding(self, node: MemoryNode) -> List[float]:
        if node.embedding is None:
            node.embedding = self.embedder.embed(node.name)
        return node.embedding

    def _get_child_embedding(self, node: MemoryNode) -> List[float]:
        if node.content_embedding is not None and not node.children:
            return node.content_embedding
        return self._get_embedding(node)

    def _update_leaf_content(self, leaf: MemoryNode, content_text: str | None) -> None:
        if not content_text:
            return
        content_text = content_text.strip()
        if not content_text:
            return
        embedding = self.embedder.embed(content_text)
        if leaf.content_embedding is None or len(leaf.content_embedding) != len(embedding):
            leaf.content_embedding = embedding
            leaf.content_count = 1
        else:
            count = max(leaf.content_count, 0)
            new_count = count + 1
            leaf.content_embedding = [
                (old * count + new) / new_count
                for old, new in zip(leaf.content_embedding, embedding)
            ]
            leaf.content_count = new_count
        leaf.recent_texts.append(content_text)
        max_samples = self.retrieval_config.leaf_summary_max_samples
        if max_samples > 0 and len(leaf.recent_texts) > max_samples:
            leaf.recent_texts = leaf.recent_texts[-max_samples:]

    def _select_or_create_leaf(
        self,
        category: MemoryNode,
        combined_text: str,
        leaf_label_hint: str,
        query: str,
        response_text: str,
    ) -> MemoryNode:
        leaf_label_hint = self._coarsen_leaf_label(leaf_label_hint)
        leaves = list(category.children.values())
        if not leaves:
            leaf = category.get_or_create_child(leaf_label_hint)
            if leaf.label is None:
                leaf.label = leaf_label_hint
            return leaf

        if not combined_text:
            leaf = category.get_or_create_child(leaf_label_hint)
            if leaf.label is None:
                leaf.label = leaf_label_hint
            return leaf

        content_embedding = self.embedder.embed(combined_text)
        best_score = -1.0
        best_leaf = None
        for leaf in leaves:
            leaf_embedding = self._get_child_embedding(leaf)
            score = self._cosine_similarity(content_embedding, leaf_embedding)
            if score > best_score:
                best_score = score
                best_leaf = leaf

        if best_leaf is None:
            leaf = category.get_or_create_child(leaf_label_hint)
            if leaf.label is None:
                leaf.label = leaf_label_hint
            return leaf

        if best_score >= self.retrieval_config.leaf_reuse_threshold:
            return best_leaf

        if len(leaves) < self.retrieval_config.max_leaves_per_category:
            leaf = category.get_or_create_child(leaf_label_hint)
            if leaf.label is None:
                leaf.label = leaf_label_hint
            return leaf

        if best_score < self.retrieval_config.leaf_merge_threshold:
            summary = self._summarize_leaf_label(
                category.name,
                best_leaf.label or best_leaf.name,
                best_leaf.recent_texts,
                query,
                response_text,
            )
            if summary:
                best_leaf.label = summary
        return best_leaf

    def _reuse_existing_leaf(self, combined_text: str) -> List[str] | None:
        if not combined_text:
            return None
        if not self.root.children:
            return None
        content_embedding = self.embedder.embed(combined_text)
        best_score = -1.0
        best_category = None
        best_leaf = None
        for category in self.root.children.values():
            for leaf in category.children.values():
                leaf_embedding = self._get_child_embedding(leaf)
                score = self._cosine_similarity(content_embedding, leaf_embedding)
                if score > best_score:
                    best_score = score
                    best_category = category
                    best_leaf = leaf
        if best_leaf is None or best_category is None:
            return None
        if best_score >= self.retrieval_config.leaf_reuse_threshold:
            return [best_category.name, best_leaf.name]
        return None

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _weighted_merge(
        prefs: List[PreferenceFusion],
        weights: List[float],
    ) -> PreferenceFusion:
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(prefs)] * len(prefs)
        else:
            weights = [w / total for w in weights]

        length = sum(w * pref.length for w, pref in zip(weights, prefs))
        density = sum(w * pref.density for w, pref in zip(weights, prefs))
        formality = sum(w * pref.formality for w, pref in zip(weights, prefs))

        tone_probs = HierarchicalMemoryTree._merge_probs([p.tone[1] for p in prefs], weights)
        emotion_probs = HierarchicalMemoryTree._merge_probs([p.emotion[1] for p in prefs], weights)

        tone_idx = int(max(range(len(tone_probs)), key=lambda i: tone_probs[i])) if tone_probs else 0
        emotion_idx = (
            int(max(range(len(emotion_probs)), key=lambda i: emotion_probs[i])) if emotion_probs else 0
        )

        return PreferenceFusion(
            tone=(tone_idx, tone_probs),
            emotion=(emotion_idx, emotion_probs),
            length=length,
            density=density,
            formality=formality,
        )

    @staticmethod
    def _merge_probs(prob_sets: List[List[float]], weights: List[float]) -> List[float]:
        if not prob_sets:
            return []
        size = len(prob_sets[0])
        merged = [0.0] * size
        for probs, weight in zip(prob_sets, weights):
            for idx, value in enumerate(probs[:size]):
                merged[idx] += weight * value
        total = sum(merged)
        if total <= 0:
            return merged
        return [value / total for value in merged]

    def _node_snapshot(self, node: MemoryNode, path: List[str]) -> Dict[str, Any]:
        fusion = self.updater.current_fusion(node.state)
        sw = self._current_sw(node.state)
        ema = self._current_ema(node.state)
        fusion_payload = self._fusion_payload(fusion)
        display_name = node.label or node.name
        return {
            "name": display_name,
            "path": path,
            "has_data": node.state.has_data(),
            "fusion": fusion_payload,
            "sw": self._fusion_payload(sw),
            "ema": self._fusion_payload(ema),
            "children": [
                self._node_snapshot(child, path + [child.name]) for child in node.children.values()
            ],
        }

    def _node_to_dict(self, node: MemoryNode) -> Dict[str, Any]:
        return {
            "name": node.name,
            "label": node.label,
            "state": self._state_to_dict(node.state),
            "embedding": node.embedding,
            "content_embedding": node.content_embedding,
            "content_count": node.content_count,
            "recent_texts": node.recent_texts,
            "children": [self._node_to_dict(child) for child in node.children.values()],
        }

    def _node_from_dict(self, payload: Dict[str, Any]) -> MemoryNode:
        name = payload.get("name", "node")
        node = MemoryNode(
            name=name,
            label=payload.get("label") or name,
            embedding=payload.get("embedding"),
            content_embedding=payload.get("content_embedding"),
            content_count=int(payload.get("content_count", 0)),
            recent_texts=list(payload.get("recent_texts") or []),
        )
        node.state = self._state_from_dict(payload.get("state", {}))
        for child_payload in payload.get("children", []):
            child = self._node_from_dict(child_payload)
            node.children[child.name] = child
        return node

    @staticmethod
    def _state_to_dict(state: PAMTState) -> Dict[str, Any]:
        return {
            "continuous_history": state.continuous_history,
            "categorical_history": state.categorical_history,
            "ema_continuous": state.ema_continuous,
            "ema_categorical": state.ema_categorical,
            "sw_history": state.sw_history,
            "ema_history": state.ema_history,
        }

    @staticmethod
    def _state_from_dict(payload: Dict[str, Any]) -> PAMTState:
        state = PAMTState()
        if not isinstance(payload, dict):
            return state
        state.continuous_history = payload.get("continuous_history", state.continuous_history)
        state.categorical_history = payload.get("categorical_history", state.categorical_history)
        state.ema_continuous = payload.get("ema_continuous", state.ema_continuous)
        state.ema_categorical = payload.get("ema_categorical", state.ema_categorical)
        state.sw_history = payload.get("sw_history", state.sw_history)
        state.ema_history = payload.get("ema_history", state.ema_history)
        return state

    @staticmethod
    def _fusion_payload(fusion: PreferenceFusion | None) -> Dict[str, Any] | None:
        if fusion is None:
            return None
        return {
            "length": fusion.length,
            "density": fusion.density,
            "formality": fusion.formality,
            "tone": {"index": fusion.tone[0], "probs": fusion.tone[1]},
            "emotion": {"index": fusion.emotion[0], "probs": fusion.emotion[1]},
        }

    @staticmethod
    def _change_payload(change: Any) -> Dict[str, Any] | None:
        if change is None:
            return None
        return {
            "scores": dict(change.scores),
            "triggered": dict(change.triggered),
            "overall_triggered": change.overall_triggered,
        }

    def _current_sw(self, state: PAMTState) -> PreferenceFusion | None:
        if not state.has_data():
            return None
        cont: Dict[str, float] = {}
        for dim in ("length", "density", "formality"):
            history = state.continuous_history[dim]
            window = history[-self.update_config.window_size :]
            cont[dim] = sum(window) / max(len(window), 1)
        cat: Dict[str, CategoryWithProb] = {}
        for dim in ("tone", "emotion"):
            history = state.categorical_history[dim]
            window = history[-self.update_config.window_size :]
            sw = PAMTUpdater._mean_vector(window)
            sw = PAMTUpdater._normalize_probs(sw)
            idx = int(max(range(len(sw)), key=lambda i: sw[i])) if sw else 0
            cat[dim] = (idx, sw)
        return PreferenceFusion(
            tone=cat["tone"],
            emotion=cat["emotion"],
            length=cont["length"],
            density=cont["density"],
            formality=cont["formality"],
        )

    def _current_ema(self, state: PAMTState) -> PreferenceFusion | None:
        if not state.has_data():
            return None
        cont: Dict[str, float] = {}
        for dim in ("length", "density", "formality"):
            cont[dim] = state.ema_continuous.get(dim, 0.0)
        cat: Dict[str, CategoryWithProb] = {}
        for dim in ("tone", "emotion"):
            ema = state.ema_categorical.get(dim) or []
            ema = PAMTUpdater._normalize_probs(ema)
            idx = int(max(range(len(ema)), key=lambda i: ema[i])) if ema else 0
            cat[dim] = (idx, ema)
        return PreferenceFusion(
            tone=cat["tone"],
            emotion=cat["emotion"],
            length=cont["length"],
            density=cont["density"],
            formality=cont["formality"],
        )



