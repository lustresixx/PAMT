from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_TONE_MODEL_ID = "FacebookAI/roberta-large-mnli"
DEFAULT_EMOTION_MODEL_ID = "skep_ernie_1.0_large_ch"
DEFAULT_FORMALITY_MODEL_ID = "s-nlp/roberta-base-formality-ranker"
DEFAULT_DENSITY_MODEL_ID = "wiki80_bert_softmax"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_TONE_MAX_LENGTH = 256
DEFAULT_FORMALITY_MAX_LENGTH = 256
DEFAULT_OPENNRE_MAX_PAIRS = 96
DEFAULT_OPENNRE_MAX_ENTITIES = 40
DEFAULT_OPENNRE_MAX_TEXT_TOKENS = 512


@dataclass
class UpdateConfig:
    window_size: int = 3
    ema_decay: float = 0.7  # EMA smoothing factor
    fuse_weight: float = 0.6  # SW/EMA blend weight
    change_threshold: float = 0.9  # change trigger threshold
    variance_window: int = 8
    epsilon: float = 1e-5


@dataclass
class PreferenceModelConfig:
    tone_model_id: str = DEFAULT_TONE_MODEL_ID
    emotion_model_id: str = DEFAULT_EMOTION_MODEL_ID
    formality_model_id: str = DEFAULT_FORMALITY_MODEL_ID
    density_model_id: str = DEFAULT_DENSITY_MODEL_ID
    spacy_model: str = DEFAULT_SPACY_MODEL
    tone_max_length: int = DEFAULT_TONE_MAX_LENGTH
    formality_max_length: int = DEFAULT_FORMALITY_MAX_LENGTH
    opennre_max_pairs: int = DEFAULT_OPENNRE_MAX_PAIRS
    opennre_max_entities: int = DEFAULT_OPENNRE_MAX_ENTITIES
    opennre_max_text_tokens: int = DEFAULT_OPENNRE_MAX_TEXT_TOKENS


@dataclass
class PreferenceConfig:
    length_history: int = 3
    length_normalizer: int = 360
    preference_models: PreferenceModelConfig = field(default_factory=PreferenceModelConfig)
    tone_labels: List[str] = field(default_factory=lambda: ["humorous", "serious", "gentle", "neutral"])
    emotion_labels: List[str] = field(default_factory=lambda: ["neutral", "joy", "sadness", "anger", "fear"])
    density_bins: List[float] = field(default_factory=lambda: [0.3, 0.55])
    length_bins: List[float] = field(default_factory=lambda: [0.28, 0.65])
    formality_bins: List[float] = field(default_factory=lambda: [0.35, 0.75])


@dataclass
class ModelConfig:
    """DeepSeek (OpenAI-compatible) settings."""
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    request_timeout: int = 600
    ollama_url: str = "http://localhost:11434/api/generate"


@dataclass
class EmbeddingConfig:
    """HF local embedding settings (single embedding model)."""
    provider: str = "hf"
    model_name: str = "facebook/contriever"
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    ollama_url: str = "http://localhost:11434/api/embeddings"
    hf_device: str | None = None  # e.g. "cpu", "cuda", "cuda:0"
    hf_max_length: int = 256
    hf_normalize: bool = True
    hf_cache_dir: str | None = None


@dataclass
class ExperimentConfig:
    dataset_path: str
    output_path: str = "results.jsonl"
    task_type: str | None = None  # "SH", "MH", "T"
    max_turns: int | None = None
    seed: int = 42
