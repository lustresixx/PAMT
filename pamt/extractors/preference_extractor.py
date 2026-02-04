from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, List, Optional, Tuple

from ..config import PreferenceConfig
from ..core.types import CategoryWithProb, PreferenceVector


ToneModel = Callable[[str], CategoryWithProb]
EmotionModel = Callable[[str], CategoryWithProb]
FormalityModel = Callable[[str], float]
OpenIEModel = Callable[[str], List[Tuple[str, str, str]]]

logger = logging.getLogger(__name__)


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
        tone_model_id: str = "FacebookAI/roberta-large-mnli",
        emotion_model_id: str = "skep_ernie_1.0_large_ch",
        formality_model_id: str = "s-nlp/roberta-base-formality-ranker",
        density_model_id: str = "wiki80_bert_softmax",
        device: int | str | None = None,
        max_length: int = 256,
        opennre_max_pairs: int = 64,
        opennre_max_entities: int = 32,
        spacy_model: str = "en_core_web_sm",
        hf_cache_dir: str | None = None,
    ) -> "ModelPreferenceExtractor":
        from .preference_models import (
            build_opennre_openie_model,
            build_roberta_tone_model,
            build_skep_emotion_model,
            build_hf_formality_model,
        )

        logger.info(
            "preference_models: tone=%s emotion=%s formality=%s density=%s device=%s",
            tone_model_id,
            emotion_model_id,
            formality_model_id,
            density_model_id,
            device,
        )

        tone_model = build_roberta_tone_model(
            config,
            model_id=tone_model_id,
            device=device,
            max_length=max_length,
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
            max_length=max_length,
            cache_dir=hf_cache_dir,
        )
        openie_model = build_opennre_openie_model(
            model_id=density_model_id,
            device=device,
            max_pairs=opennre_max_pairs,
            max_entities=opennre_max_entities,
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
        return min(avg_len / 300.0, 1.0)

    def _information_density(self, assistant_text: str) -> float:
        triples = self.openie_model(assistant_text)
        words = assistant_text.split()
        if not words:
            return 0.0
        return min(len(triples) / max(len(words), 1), 1.0)


def build_preference_extractor(
    config: PreferenceConfig,
    *,
    prefer_model: bool = True,
) -> PreferenceExtractor:
    if not prefer_model:
        raise ValueError("prefer_model=False is not supported in strict model-only mode.")
    return ModelPreferenceExtractor.from_preferred_models(config)
