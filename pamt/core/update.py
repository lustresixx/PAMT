from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import UpdateConfig
from .types import CategoryWithProb, ChangeSignal, PreferenceFusion, PreferenceVector


ContinuousDim = str
CategoricalDim = str
logger = logging.getLogger(__name__)


@dataclass
class PAMTState:
    continuous_history: Dict[ContinuousDim, List[float]] = field(
        default_factory=lambda: {"length": [], "density": [], "formality": []}
    )
    categorical_history: Dict[CategoricalDim, List[List[float]]] = field(
        default_factory=lambda: {"tone": [], "emotion": []}
    )
    ema_continuous: Dict[ContinuousDim, float] = field(
        default_factory=lambda: {"length": 0.0, "density": 0.0, "formality": 0.0}
    )
    ema_categorical: Dict[CategoricalDim, List[float]] = field(
        default_factory=lambda: {"tone": [], "emotion": []}
    )
    sw_history: Dict[str, List[float]] = field(
        default_factory=lambda: {"length": [], "density": [], "formality": [], "tone": [], "emotion": []}
    )
    ema_history: Dict[str, List[float]] = field(
        default_factory=lambda: {"length": [], "density": [], "formality": [], "tone": [], "emotion": []}
    )

    def has_data(self) -> bool:
        """Return True when every dimension has at least one observation."""
        has_continuous = all(self.continuous_history[dim] for dim in self.continuous_history)
        has_categorical = all(self.categorical_history[dim] for dim in self.categorical_history)
        return has_continuous and has_categorical


class PAMTUpdater:
    """Implements preference fusion and change detection."""
    def __init__(self, config: UpdateConfig):
        self.config = config

    def update(
        self,
        state: PAMTState,
        pref: PreferenceVector,
        weight: float = 1.0,
    ) -> tuple[PreferenceFusion, ChangeSignal]:
        # Update continuous and categorical dimensions separately, then fuse.
        weight = self._clamp_weight(weight)
        cont = self._update_continuous(state, pref, weight)
        cat = self._update_categorical(state, pref, weight)
        fusion = PreferenceFusion(
            tone=cat["tone"],
            emotion=cat["emotion"],
            length=cont["length"],
            density=cont["density"],
            formality=cont["formality"],
        )
        change = self._change_signal(state)
        dims = ["length", "density", "formality", "tone", "emotion"]
        sw_last = {dim: self._last_value(state.sw_history, dim) for dim in dims}
        ema_last = {dim: self._last_value(state.ema_history, dim) for dim in dims}
        logger.info(
            "PAMTUpdater.update: weight=%.3f pref(tone=%s emo=%s len=%.3f den=%.3f form=%.3f) "
            "fusion(len=%.3f den=%.3f form=%.3f) "
            "sw(%s) ema(%s) scores(%s) change=%s",
            weight,
            pref.tone[0] if pref.tone else None,
            pref.emotion[0] if pref.emotion else None,
            pref.length,
            pref.density,
            pref.formality,
            fusion.length,
            fusion.density,
            fusion.formality,
            self._format_dims(sw_last),
            self._format_dims(ema_last),
            self._format_dims(change.scores),
            change.overall_triggered,
        )
        if change.overall_triggered:
            logger.debug("PAMTUpdater.update: change triggered %s", change.triggered)
        return fusion, change

    def _update_continuous(self, state: PAMTState, pref: PreferenceVector, weight: float) -> Dict[str, float]:
        updated: Dict[str, float] = {}
        for dim, value in [("length", pref.length), ("density", pref.density), ("formality", pref.formality)]:
            value = self._apply_continuous_weight(state, dim, value, weight)
            # Track raw preference values for sliding-window statistics.
            history = state.continuous_history[dim]
            history.append(value)
            window = history[-self.config.window_size :]
            # Sliding Window (SW) average for short-term preference changes.
            sw = sum(window) / max(len(window), 1)
            # Exponential Moving Average (EMA) for long-term preference trends.
            prev_ema = state.ema_continuous.get(dim, value)
            ema = self.config.ema_decay * prev_ema + (1 - self.config.ema_decay) * value
            state.ema_continuous[dim] = ema
            # Fused perception vector w_t(d) = lambda * SW + (1-lambda) * EMA.
            fused = self.config.fuse_weight * sw + (1 - self.config.fuse_weight) * ema
            updated[dim] = fused
            # Cache SW/EMA histories for change detection variance.
            self._track_variance(state, dim, sw, ema)
        return updated

    def _update_categorical(
        self,
        state: PAMTState,
        pref: PreferenceVector,
        weight: float,
    ) -> Dict[str, CategoryWithProb]:
        updated: Dict[str, CategoryWithProb] = {}
        for dim, value in [("tone", pref.tone), ("emotion", pref.emotion)]:
            _, probs = value
            probs = self._normalize_probs(probs)
            probs = self._apply_categorical_weight(state, dim, probs, weight)
            history = state.categorical_history[dim]
            history.append(probs)
            window = history[-self.config.window_size :]
            # SW/EMA operate on probability vectors for categorical dimensions.
            sw = self._mean_vector(window)
            prev_ema = state.ema_categorical.get(dim) or probs
            ema = [
                self.config.ema_decay * prev + (1 - self.config.ema_decay) * cur
                for prev, cur in zip(prev_ema, probs)
            ]
            state.ema_categorical[dim] = ema
            fused = [
                self.config.fuse_weight * s + (1 - self.config.fuse_weight) * e
                for s, e in zip(sw, ema)
            ]
            fused = self._normalize_probs(fused)
            idx = int(max(range(len(fused)), key=lambda i: fused[i])) if fused else 0
            updated[dim] = (idx, fused)
            # Use vector variance as a scalar summary for change detection.
            self._track_variance(state, dim, self._vector_variance(sw), self._vector_variance(ema))
        return updated

    def current_fusion(self, state: PAMTState) -> PreferenceFusion | None:
        """Compute the current fused preference without mutating state."""
        if not state.has_data():
            return None
        cont: Dict[str, float] = {}
        for dim in ("length", "density", "formality"):
            history = state.continuous_history[dim]
            window = history[-self.config.window_size :]
            sw = sum(window) / max(len(window), 1)
            ema = state.ema_continuous.get(dim, sw)
            cont[dim] = self.config.fuse_weight * sw + (1 - self.config.fuse_weight) * ema
        cat: Dict[str, CategoryWithProb] = {}
        for dim in ("tone", "emotion"):
            history = state.categorical_history[dim]
            window = history[-self.config.window_size :]
            sw = self._mean_vector(window)
            ema = state.ema_categorical.get(dim) or sw
            fused = [
                self.config.fuse_weight * s + (1 - self.config.fuse_weight) * e
                for s, e in zip(sw, ema)
            ]
            fused = self._normalize_probs(fused)
            idx = int(max(range(len(fused)), key=lambda i: fused[i])) if fused else 0
            cat[dim] = (idx, fused)
        return PreferenceFusion(
            tone=cat["tone"],
            emotion=cat["emotion"],
            length=cont["length"],
            density=cont["density"],
            formality=cont["formality"],
        )

    def _change_signal(self, state: PAMTState) -> ChangeSignal:
        scores: Dict[str, float] = {}
        triggered: Dict[str, bool] = {}
        for dim in ["length", "density", "formality", "tone", "emotion"]:
            sw_hist = state.sw_history[dim][-self.config.variance_window :]
            ema_hist = state.ema_history[dim][-self.config.variance_window :]
            if not sw_hist or not ema_hist:
                scores[dim] = 0.0
                triggered[dim] = False
                continue
            # Delta_t(d) = |SW_t(d) - EMA_t(d)|
            delta = abs(sw_hist[-1] - ema_hist[-1])
            # C_t(d) = Delta / (epsilon + Var(SW) + Var(EMA))
            var_sw = self._variance(sw_hist)
            var_ema = self._variance(ema_hist)
            score = delta / (self.config.epsilon + var_sw + var_ema)
            scores[dim] = score
            triggered[dim] = score > self.config.change_threshold
        overall_triggered = any(triggered.values())
        return ChangeSignal(scores=scores, triggered=triggered, overall_triggered=overall_triggered)

    def _track_variance(self, state: PAMTState, dim: str, sw_value: float, ema_value: float) -> None:
        state.sw_history[dim].append(sw_value)
        state.ema_history[dim].append(ema_value)

    @staticmethod
    def _last_value(history: Dict[str, List[float]], dim: str) -> float | None:
        values = history.get(dim)
        if not values:
            return None
        return values[-1]

    @staticmethod
    def _format_dims(values: Dict[str, float | None]) -> str:
        def _fmt(val: float | None) -> str:
            if val is None:
                return "None"
            return f"{val:.3f}"

        parts = [f"{key}={_fmt(values.get(key))}" for key in ["length", "density", "formality", "tone", "emotion"]]
        return " ".join(parts)

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight

    @staticmethod
    def _apply_continuous_weight(
        state: PAMTState,
        dim: str,
        value: float,
        weight: float,
    ) -> float:
        if weight >= 1.0:
            return value
        prev = state.ema_continuous.get(dim, value)
        return prev * (1 - weight) + value * weight

    @staticmethod
    def _apply_categorical_weight(
        state: PAMTState,
        dim: str,
        probs: List[float],
        weight: float,
    ) -> List[float]:
        if weight >= 1.0:
            return probs
        prev = state.ema_categorical.get(dim)
        if not prev:
            return probs
        blended = [(1 - weight) * p + weight * q for p, q in zip(prev, probs)]
        total = sum(blended)
        if total <= 0:
            return blended
        return [p / total for p in blended]

    @staticmethod
    def _mean_vector(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        size = len(vectors[0])
        acc = [0.0] * size
        for vec in vectors:
            for i, val in enumerate(vec):
                acc[i] += val
        return [val / max(len(vectors), 1) for val in acc]

    @staticmethod
    def _normalize_probs(probs: List[float]) -> List[float]:
        total = sum(probs)
        if total <= 0:
            return probs
        return [p / total for p in probs]

    @staticmethod
    def _variance(values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    @staticmethod
    def _vector_variance(vec: List[float]) -> float:
        if not vec:
            return 0.0
        mean = sum(vec) / len(vec)
        return sum((v - mean) ** 2 for v in vec) / len(vec)



