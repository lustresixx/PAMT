from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Callable, List, Optional, Tuple

from ..config import PreferenceConfig, PreferenceModelConfig
from ..core.types import CategoryWithProb, PreferenceVector


ToneModel = Callable[[str], CategoryWithProb]
EmotionModel = Callable[[str], CategoryWithProb]
FormalityModel = Callable[[str], float]
OpenIEModel = Callable[[str], List[Tuple[str, str, str]]]

logger = logging.getLogger(__name__)


_DENSITY_TOKEN_RE = re.compile("[A-Za-z]+(?:'[A-Za-z]+)?|\\d+(?:\\.\\d+)?|[\u4e00-\u9fff]+")
_DENSITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "might",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "will",
    "with",
    "would",
    "you",
    "your",
}


@dataclass
class ExtractionState:
    response_lengths: List[int] = field(default_factory=list)


class PreferenceExtractor:
    """Base interface for extracting preference vectors from dialogue turns."""
    def extract(
        self,
        user_text: str,
        assistant_text: str,
        state: ExtractionState,
        assistant_token_count: int | None = None,
    ) -> PreferenceVector:
        raise NotImplementedError


class ModelPreferenceExtractor(PreferenceExtractor):
    """Model-backed extractor using the paper-specified models."""
    def __init__(
        self,
        config: PreferenceConfig,
        tone_model: ToneModel,
        emotion_model: EmotionModel,
        formality_model: FormalityModel,
        openie_model: OpenIEModel,
    ):
        self.config = config
        self.tone_model = tone_model
        self.emotion_model = emotion_model
        self.formality_model = formality_model
        self.openie_model = openie_model

    @classmethod
    def from_preferred_models(
        cls,
        config: PreferenceConfig,
        *,
        model_config: PreferenceModelConfig | None = None,
        tone_model_id: str | None = None,
        emotion_model_id: str | None = None,
        formality_model_id: str | None = None,
        density_model_id: str | None = None,
        device: int | str | None = None,
        max_length: int | None = None,
        opennre_max_pairs: int | None = None,
        opennre_max_entities: int | None = None,
        opennre_max_text_tokens: int | None = None,
        spacy_model: str | None = None,
        hf_cache_dir: str | None = None,
    ) -> "ModelPreferenceExtractor":
        from .preference_models import (
            build_opennre_openie_model,
            build_roberta_tone_model,
            build_skep_emotion_model,
            build_hf_formality_model,
        )

        model_cfg = model_config or config.preference_models
        tone_model_id = tone_model_id or model_cfg.tone_model_id
        emotion_model_id = emotion_model_id or model_cfg.emotion_model_id
        formality_model_id = formality_model_id or model_cfg.formality_model_id
        density_model_id = density_model_id or model_cfg.density_model_id
        spacy_model = spacy_model or model_cfg.spacy_model
        opennre_max_pairs = (
            opennre_max_pairs
            if opennre_max_pairs is not None
            else model_cfg.opennre_max_pairs
        )
        opennre_max_entities = (
            opennre_max_entities
            if opennre_max_entities is not None
            else model_cfg.opennre_max_entities
        )
        opennre_max_text_tokens = (
            opennre_max_text_tokens
            if opennre_max_text_tokens is not None
            else model_cfg.opennre_max_text_tokens
        )
        tone_max_length = (
            max_length if max_length is not None else model_cfg.tone_max_length
        )
        formality_max_length = (
            max_length if max_length is not None else model_cfg.formality_max_length
        )

        logger.info(
            "preference_models: tone=%s emotion=%s formality=%s density=%s spacy=%s device=%s",
            tone_model_id,
            emotion_model_id,
            formality_model_id,
            density_model_id,
            spacy_model,
            device,
        )

        tone_model = build_roberta_tone_model(
            config,
            model_id=tone_model_id,
            device=device,
            max_length=tone_max_length,
            cache_dir=hf_cache_dir,
        )
        emotion_model = build_skep_emotion_model(
            config,
            model_id=emotion_model_id,
            device=device,
        )
        formality_model = build_hf_formality_model(
            model_id=formality_model_id,
            device=device,
            max_length=formality_max_length,
            cache_dir=hf_cache_dir,
        )
        openie_model = build_opennre_openie_model(
            model_id=density_model_id,
            device=device,
            max_pairs=opennre_max_pairs,
            max_entities=opennre_max_entities,
            max_text_tokens=opennre_max_text_tokens,
            spacy_model=spacy_model,
        )
        return cls(
            config=config,
            tone_model=tone_model,
            emotion_model=emotion_model,
            formality_model=formality_model,
            openie_model=openie_model,
        )

    def extract(
        self,
        user_text: str,
        assistant_text: str,
        state: ExtractionState,
        assistant_token_count: int | None = None,
    ) -> PreferenceVector:
        tone = self.tone_model(user_text)
        emotion = self.emotion_model(user_text + "\n" + assistant_text)
        length = self._response_length(assistant_text, state, assistant_token_count)
        density = self._information_density(assistant_text)
        formality = self.formality_model(assistant_text)
        return PreferenceVector(
            tone=tone,
            emotion=emotion,
            length=length,
            density=density,
            formality=formality,
        )

    def _response_length(
        self,
        assistant_text: str,
        state: ExtractionState,
        assistant_token_count: int | None = None,
    ) -> float:
        token_count = assistant_token_count
        if token_count is None:
            token_count = len(assistant_text.split())
        token_count = max(int(token_count), 0)
        state.response_lengths.append(token_count)
        history = state.response_lengths[-self.config.length_history :]
        if not history:
            return 0.0
        avg_len = sum(history) / len(history)
        normalizer = max(int(self.config.length_normalizer), 1)
        return min(avg_len / float(normalizer), 1.0)

    def _information_density(self, assistant_text: str) -> float:
        if not assistant_text.strip():
            return 0.0
        triples = self.openie_model(assistant_text)
        tokens, content_count, numeric_count, cjk_count = _density_stats(assistant_text)
        token_count = len(tokens)
        if token_count <= 0:
            token_count = len(assistant_text.split())
        if token_count <= 0:
            return 0.0
        triple_density = min(len(triples) / float(token_count), 1.0)
        lexical_density = _lexical_density_from_stats(
            tokens,
            content_count,
            numeric_count,
            cjk_count,
        )
        if not triples:
            return lexical_density
        return min((triple_density * 0.5) + (lexical_density * 0.5), 1.0)


def build_preference_extractor(
    config: PreferenceConfig,
    *,
    prefer_model: bool = True,
) -> PreferenceExtractor:
    if not prefer_model:
        raise ValueError("prefer_model=False is not supported in strict model-only mode.")
    return ModelPreferenceExtractor.from_preferred_models(config)


def _normalize_density_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("**", " ")
        .replace("__", " ")
        .replace("`", " ")
        .replace("*", " ")
        .replace("_", " ")
    )


def _is_cjk(token: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in token)


def _chunk_cjk(token: str, size: int = 2) -> List[str]:
    if len(token) <= size:
        return [token]
    return [token[i : i + size] for i in range(0, len(token), size)]


def _density_stats(text: str) -> Tuple[List[str], int, int, int]:
    cleaned = _normalize_density_text(text)
    raw_tokens = _DENSITY_TOKEN_RE.findall(cleaned)
    tokens: List[str] = []
    content_count = 0
    numeric_count = 0
    cjk_count = 0
    for token in raw_tokens:
        if _is_cjk(token):
            chunks = _chunk_cjk(token)
            tokens.extend(chunks)
            content_count += len(chunks)
            cjk_count += len(chunks)
            continue
        norm = token.lower()
        tokens.append(norm)
        if norm not in _DENSITY_STOPWORDS:
            content_count += 1
        if any(ch.isdigit() for ch in norm):
            numeric_count += 1
    return tokens, content_count, numeric_count, cjk_count


def _lexical_density_from_stats(
    tokens: List[str],
    content_count: int,
    numeric_count: int,
    cjk_count: int,
) -> float:
    total = len(tokens)
    if total == 0:
        return 0.0
    unique_ratio = len(set(tokens)) / total
    content_ratio = content_count / total
    numeric_ratio = numeric_count / total
    density = (0.45 * content_ratio) + (0.4 * unique_ratio) + (0.15 * numeric_ratio)
    if (cjk_count / total) >= 0.6:
        density *= 0.7
    return _clamp(density)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
