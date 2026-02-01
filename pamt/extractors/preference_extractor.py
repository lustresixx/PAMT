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

# This module contains both mock (heuristic) and real (model-backed) extractors.

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


class HeuristicPreferenceExtractor(PreferenceExtractor):
    """Mock/heuristic extractor that approximates preference signals without real models."""
    def __init__(self, config: PreferenceConfig):
        self.config = config

    def extract(
        self,
        user_text: str,
        assistant_text: str,
        state: ExtractionState,
        assistant_token_count: int | None = None,
    ) -> PreferenceVector:
        # Mock signals based on simple keyword rules (fast, no external models).
        tone = self._infer_tone(user_text)
        emotion = self._infer_emotion(user_text)
        length = self._response_length(assistant_text, state, assistant_token_count)
        # Mock density/formality derived from punctuation and contractions.
        density = self._information_density(assistant_text)
        formality = self._formality_score(assistant_text)
        return PreferenceVector(
            tone=tone,
            emotion=emotion,
            length=length,
            density=density,
            formality=formality,
        )

    def _infer_tone(self, text: str) -> CategoryWithProb:
        text_lower = text.lower()
        labels = self.config.tone_labels
        if "!" in text or "great" in text_lower or "thanks" in text_lower:
            idx = labels.index("friendly") if "friendly" in labels else 0
        elif "please" in text_lower or "regards" in text_lower:
            idx = labels.index("formal") if "formal" in labels else 0
        else:
            idx = labels.index("neutral") if "neutral" in labels else 0
        probs = [0.0] * len(labels)
        probs[idx] = 1.0
        return idx, probs

    def _infer_emotion(self, text: str) -> CategoryWithProb:
        text_lower = text.lower()
        labels = self.config.emotion_labels
        if any(w in text_lower for w in ["happy", "glad", "excited"]):
            idx = labels.index("joy") if "joy" in labels else 0
        elif any(w in text_lower for w in ["sad", "down", "unhappy"]):
            idx = labels.index("sadness") if "sadness" in labels else 0
        elif any(w in text_lower for w in ["angry", "mad", "upset"]):
            idx = labels.index("anger") if "anger" in labels else 0
        else:
            idx = labels.index("neutral") if "neutral" in labels else 0
        probs = [0.0] * len(labels)
        probs[idx] = 1.0
        return idx, probs

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
        # Normalize to [0, 1] with a soft cap at 300 tokens.
        return min(avg_len / 300.0, 1.0)

    def _information_density(self, assistant_text: str) -> float:
        words = assistant_text.split()
        if not words:
            return 0.0
        # Rough proxy for OpenIE triple density: punctuation often separates facts.
        triple_count = max(1, sum(assistant_text.count(p) for p in [".", ";", ":"]))
        return min(triple_count / max(len(words), 1), 1.0)

    def _formality_score(self, assistant_text: str) -> float:
        words = assistant_text.split()
        if not words:
            return 0.0
        # Heuristic: fewer contractions -> more formal.
        contractions = ["n't", "'re", "'ll", "'d", "'ve", "'m"]
        contraction_hits = sum(1 for w in words if any(c in w for c in contractions))
        score = 1.0 - min(contraction_hits / max(len(words), 1), 1.0)
        return max(min(score, 1.0), 0.0)


class ModelPreferenceExtractor(PreferenceExtractor):
    """Real model-backed extractor that plugs in classifiers and OpenIE-style models."""
    def __init__(
        self,
        config: PreferenceConfig,
        tone_model: ToneModel,
        emotion_model: EmotionModel,
        formality_model: FormalityModel,
        openie_model: Optional[OpenIEModel] = None,
    ):
        self.config = config
        self.tone_model = tone_model
        self.emotion_model = emotion_model
        self.formality_model = formality_model
        self.openie_model = openie_model

    @classmethod
    def from_huggingface(
        cls,
        config: PreferenceConfig,
        *,
        tone_model_id: str = "cardiffnlp/twitter-roberta-base-irony",
        emotion_model_id: str = "SamLowe/roberta-base-go_emotions",
        formality_model_id: str = "s-nlp/roberta-base-formality-ranker",
        density_model_id: str | None = "urchade/gliner_small-v2.1",
        device: int | str | None = None,
        max_length: int = 256,
        gliner_labels: list[str] | None = None,
        gliner_threshold: float = 0.5,
    ) -> "ModelPreferenceExtractor":
        from .preference_models import (
            build_gliner_openie_model,
            build_hf_emotion_model,
            build_hf_formality_model,
            build_hf_tone_model,
        )

        # Real models: HuggingFace classifiers + GLiNER (entity extraction) for density.
        tone_model = build_hf_tone_model(
            config,
            model_id=tone_model_id,
            device=device,
            max_length=max_length,
        )
        emotion_model = build_hf_emotion_model(
            config,
            model_id=emotion_model_id,
            device=device,
            max_length=max_length,
        )
        formality_model = build_hf_formality_model(
            model_id=formality_model_id,
            device=device,
            max_length=max_length,
        )
        openie_model = None
        if density_model_id:
            openie_model = build_gliner_openie_model(
                model_id=density_model_id,
                labels=gliner_labels,
                threshold=gliner_threshold,
                device=device,
            )
        return cls(
            config=config,
            tone_model=tone_model,
            emotion_model=emotion_model,
            formality_model=formality_model,
            openie_model=openie_model,
        )

    @classmethod
    def from_preferred_models(
        cls,
        config: PreferenceConfig,
        *,
        tone_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model_id: str = "skep_ernie_1.0_large_ch",
        formality_model_id: str = "s-nlp/roberta-base-formality-ranker",
        density_model_id: str | None = "wiki80_bert_softmax",
        device: int | str | None = None,
        max_length: int = 256,
        opennre_max_pairs: int = 64,
        opennre_max_entities: int = 32,
        spacy_model: str = "en_core_web_sm",
    ) -> "ModelPreferenceExtractor":
        from .preference_models import (
            build_opennre_openie_model,
            build_roberta_tone_model,
            build_skep_emotion_model,
            build_hf_formality_model,
        )

        tone_model = build_roberta_tone_model(
            config,
            model_id=tone_model_id,
            device=device,
            max_length=max_length,
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
        )
        openie_model = None
        if density_model_id:
            openie_model = build_opennre_openie_model(
                model_id=density_model_id,
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
        if not self.openie_model:
            return 0.0
        triples = self.openie_model(assistant_text)
        words = assistant_text.split()
        if not words:
            return 0.0
        return min(len(triples) / max(len(words), 1), 1.0)


class FallbackPreferenceExtractor(PreferenceExtractor):
    """Try a primary extractor, fallback to heuristic when it fails."""

    def __init__(self, primary: PreferenceExtractor, fallback: PreferenceExtractor):
        self.primary = primary
        self.fallback = fallback
        self._use_primary = True

    def extract(
        self,
        user_text: str,
        assistant_text: str,
        state: ExtractionState,
        assistant_token_count: int | None = None,
    ) -> PreferenceVector:
        if self._use_primary:
            try:
                return self.primary.extract(
                    user_text,
                    assistant_text,
                    state,
                    assistant_token_count,
                )
            except Exception as exc:
                logger.warning("Model extractor failed, falling back: %s", exc)
                self._use_primary = False
        return self.fallback.extract(
            user_text,
            assistant_text,
            state,
            assistant_token_count,
        )


def build_preference_extractor(
    config: PreferenceConfig,
    *,
    prefer_model: bool = True,
) -> PreferenceExtractor:
    fallback = HeuristicPreferenceExtractor(config)
    if not prefer_model:
        return fallback
    try:
        primary = ModelPreferenceExtractor.from_preferred_models(config)
    except Exception as exc:
        logger.warning("Preferred extractor unavailable, trying HF fallback: %s", exc)
        try:
            primary = ModelPreferenceExtractor.from_huggingface(config)
        except Exception as hf_exc:
            logger.warning("Model extractor unavailable, using heuristic: %s", hf_exc)
            return fallback
    return FallbackPreferenceExtractor(primary, fallback)
