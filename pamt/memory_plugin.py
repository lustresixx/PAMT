from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import EmbeddingConfig, ModelConfig, PreferenceConfig, UpdateConfig
from .core.memory_tree import HierarchicalMemoryTree, RetrievalConfig
from .core.prompting import build_prompt
from .extractors.preference_extractor import (
    ExtractionState,
    PreferenceExtractor,
    build_preference_extractor,
)
from .embeddings.models import DeepSeekEmbeddings, EmbeddingClient, HFLocalEmbeddings, OllamaEmbeddings
from .llms.models import DeepSeekLLM, LLM, OllamaLLM


def _build_prompt_source(history: List[str], user_text: str) -> str:
    if not history:
        return user_text
    return "\n".join(history + [f"User: {user_text}"])


def _build_retrieval_source(history: List[str], user_text: str, max_turns: int = 1) -> str:
    if not history:
        return user_text
    if max_turns <= 0:
        return user_text
    take = max_turns * 2
    context = history[-take:]
    return "\n".join(context + [f"User: {user_text}"])


def _build_llm(config: ModelConfig) -> LLM:
    if config.provider == "ollama":
        return OllamaLLM(config)
    if config.provider == "deepseek":
        return DeepSeekLLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _build_embedder(config: EmbeddingConfig) -> EmbeddingClient:
    if config.provider == "ollama":
        return OllamaEmbeddings(config)
    if config.provider == "deepseek":
        return DeepSeekEmbeddings(config)
    if config.provider == "hf":
        return HFLocalEmbeddings(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


@dataclass
class MemoryAugmentation:
    prompt: str
    fusion: Any | None = None
    trace: Dict[str, Any] | None = None
    context_used: bool = False


@dataclass
class MemoryPromptPlugin:
    tree: HierarchicalMemoryTree
    pref_config: PreferenceConfig
    extractor: PreferenceExtractor
    extractor_state: ExtractionState = field(default_factory=ExtractionState)
    history_lines: List[str] = field(default_factory=list)
    last_trace: Dict[str, Any] | None = None
    storage_path: str | None = None

    def _is_short_query(self, user_text: str) -> bool:
        threshold = self.tree.retrieval_config.short_query_max_chars
        return len(user_text.strip()) <= threshold

    def _should_use_context_retrieval(self, user_text: str, trace: Dict[str, Any] | None) -> bool:
        if not self.history_lines:
            return False
        if not self._is_short_query(user_text):
            return False
        if not trace:
            return False
        return trace.get("strategy") == "fallback"

    def augment(
        self,
        user_text: str,
        *,
        use_context: bool = True,
        max_context_turns: int = 1,
    ) -> MemoryAugmentation:
        prompt_source = (
            _build_prompt_source(self.history_lines, user_text) if use_context else user_text
        )
        fusion = None
        trace = None
        context_used = False
        fusion, trace = self.tree.get_preference_trace(user_text)
        if use_context and self._should_use_context_retrieval(user_text, trace):
            retrieval_text = _build_retrieval_source(
                self.history_lines, user_text, max_turns=max_context_turns
            )
            fusion, trace = self.tree.get_preference_trace(retrieval_text)
            context_used = True
            if trace is not None:
                trace["context_used"] = True

        if fusion is None:
            prompt = prompt_source
        else:
            prompt = build_prompt(prompt_source, fusion, self.pref_config)

        self.last_trace = trace
        return MemoryAugmentation(prompt=prompt, fusion=fusion, trace=trace, context_used=context_used)

    def update(
        self,
        user_text: str,
        assistant_text: str,
        assistant_token_count: int | None = None,
    ) -> None:
        pref = self.extractor.extract(
            user_text,
            assistant_text,
            self.extractor_state,
            assistant_token_count,
        )
        path = self.tree.route_response(user_text, assistant_text)
        content_text = self.tree._combine_text(user_text, assistant_text)
        self.tree.update_preference(path, pref, content_text)
        self.history_lines.append(f"User: {user_text}")
        self.history_lines.append(f"Assistant: {assistant_text}")
        if self.storage_path:
            self.save(self.storage_path)

    def reset(self) -> None:
        self.history_lines = []
        self.last_trace = None
        if self.storage_path:
            self.save(self.storage_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history_lines": self.history_lines,
            "extractor_state": {
                "response_lengths": list(self.extractor_state.response_lengths),
            },
            "tree": self.tree.to_dict(),
        }

    def load_from_dict(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.history_lines = list(payload.get("history_lines") or [])
        state_payload = payload.get("extractor_state") or {}
        if isinstance(state_payload, dict):
            self.extractor_state.response_lengths = list(state_payload.get("response_lengths") or [])
        tree_payload = payload.get("tree")
        if isinstance(tree_payload, dict):
            self.tree.load_from_dict(tree_payload)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), ensure_ascii=True, indent=2)
        target.write_text(data, encoding="utf-8")

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Memory plugin file not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Memory plugin file must contain a JSON object.")
        self.load_from_dict(payload)

    def load_or_create(self, path: str | Path) -> None:
        target = Path(path)
        if target.is_file():
            self.load(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.save(target)


def create_memory_plugin(
    *,
    model_config: ModelConfig,
    embedding_config: EmbeddingConfig,
    preference_config: PreferenceConfig | None = None,
    update_config: UpdateConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    prefer_model_extractor: bool = True,
    label_prompt_path: str = "prompts/category_leaf.txt",
    leaf_prompt_path: str = "prompts/leaf_only.txt",
    merge_prompt_path: str = "prompts/leaf_merge_summary.txt",
    storage_path: str | None = None,
) -> MemoryPromptPlugin:
    pref_config = preference_config or PreferenceConfig()
    upd_config = update_config or UpdateConfig()
    retr_config = retrieval_config or RetrievalConfig()

    llm = _build_llm(model_config)
    embedder = _build_embedder(embedding_config)
    extractor = build_preference_extractor(pref_config, prefer_model=prefer_model_extractor)

    tree = HierarchicalMemoryTree(
        update_config=upd_config,
        retrieval_config=retr_config,
        embedder=embedder,
        llm=llm,
        label_prompt_path=label_prompt_path,
        leaf_prompt_path=leaf_prompt_path,
        merge_prompt_path=merge_prompt_path,
    )
    plugin = MemoryPromptPlugin(
        tree=tree,
        pref_config=pref_config,
        extractor=extractor,
        storage_path=storage_path,
    )
    if storage_path:
        plugin.load_or_create(storage_path)
    return plugin
