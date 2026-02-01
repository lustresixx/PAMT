from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


ProbabilityVector = List[float]
CategoryWithProb = Tuple[int, ProbabilityVector]


@dataclass
class PreferenceVector:
    # Tone and emotion are categorical (category index + prob distribution).
    tone: CategoryWithProb
    emotion: CategoryWithProb
    # Continuous dimensions are normalized to [0, 1].
    length: float
    density: float
    formality: float


@dataclass
class PreferenceFusion:
    tone: CategoryWithProb
    emotion: CategoryWithProb
    length: float
    density: float
    formality: float


@dataclass
class ChangeSignal:
    # Per-dimension change detection scores.
    scores: Dict[str, float]
    triggered: Dict[str, bool]
    overall_triggered: bool
