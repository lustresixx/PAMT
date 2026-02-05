from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UpdateConfig:
    window_size: int = 3
    ema_decay: float = 0.7  # EMA smoothing factor
    fuse_weight: float = 0.6  # SW/EMA blend weight
    change_threshold: float = 0.9  # change trigger threshold
    variance_window: int = 8
    epsilon: float = 1e-5


@dataclass
class PreferenceConfig:
    length_history: int = 3
    length_normalizer: int = 360
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
