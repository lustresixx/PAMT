from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UpdateConfig:
    window_size: int = 2
    ema_decay: float = 0.6  # EMA smoothing factor
    fuse_weight: float = 0.5  # SW/EMA blend weight
    change_threshold: float = 1.0  # change trigger threshold
    variance_window: int = 10
    epsilon: float = 1e-6


@dataclass
class PreferenceConfig:
    length_history: int = 5
    tone_labels: List[str] = field(default_factory=lambda: ["neutral", "friendly", "formal", "casual"])
    emotion_labels: List[str] = field(default_factory=lambda: ["neutral", "joy", "sadness", "anger", "fear"])
    density_bins: List[float] = field(default_factory=lambda: [0.2, 0.6])
    length_bins: List[float] = field(default_factory=lambda: [0.3, 0.6])
    formality_bins: List[float] = field(default_factory=lambda: [0.3, 0.7])


@dataclass
class ModelConfig:
    provider: str = "ollama"  # "ollama" | "deepseek" | "local"
    model_name: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    # DeepSeek (OpenAI-compatible) settings.
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    request_timeout: int = 600


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"  # "ollama" | "deepseek" | "hf"
    model_name: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    request_timeout: int = 600
    hf_device: str | None = None  # e.g. "cpu", "cuda", "cuda:0"
    hf_max_length: int = 256
    hf_normalize: bool = True


@dataclass
class ExperimentConfig:
    dataset_path: str
    output_path: str = "results.jsonl"
    task_type: str | None = None  # "SH", "MH", "T"
    max_turns: int | None = None
    seed: int = 42
