from __future__ import annotations

from ..config import PreferenceConfig
from .types import PreferenceFusion


def quantize(value: float, bins: list[float], labels: list[str]) -> str:
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return labels[i]
    return labels[-1]


def format_preference(pref: PreferenceFusion, config: PreferenceConfig) -> str:
    # Map numeric preferences to discrete labels for the control prompt.
    tone_label = _label_from_probs(pref.tone, config.tone_labels)
    emotion_label = _label_from_probs(pref.emotion, config.emotion_labels)

    length_label = quantize(
        pref.length,
        config.length_bins,
        ["short", "medium", "long"],
    )
    length_guidance = _length_guidance(length_label, config)
    density_label = quantize(
        pref.density,
        config.density_bins,
        ["sparse", "medium", "dense"],
    )
    formality_label = quantize(
        pref.formality,
        config.formality_bins,
        ["casual", "neutral", "formal"],
    )

    return (
        "If the user's query specifies content requirements, follow those; otherwise use these preferences. "
        f"Tone: {tone_label}, Emotion: {emotion_label}, "
        f"Density: {density_label}, Length: {length_guidance}, Formality: {formality_label}. "
    )


def build_prompt(user_text: str, pref: PreferenceFusion, config: PreferenceConfig) -> str:
    # Style-control prompt: "Respond in style: <prefs> + user input".
    desc = format_preference(pref, config)
    return f"Respond in style: {desc}\n\n{user_text}"


def _label_from_probs(value: tuple[int, list[float]], labels: list[str]) -> str:
    idx, _ = value
    if 0 <= idx < len(labels):
        return labels[idx]
    return labels[0] if labels else "unknown"


def _length_guidance(length_label: str, config: PreferenceConfig) -> str:
    ranges = _length_ranges(config)
    word_min, word_max = ranges.get(length_label, ranges["medium"])
    char_min = int(round(word_min * 2))
    char_max = int(round(word_max * 2))
    return (
        f"{length_label} (~{word_min}-{word_max} words, "
        f"or ~{char_min}-{char_max} Chinese characters)"
    )


def _length_ranges(config: PreferenceConfig) -> dict[str, tuple[int, int]]:
    base = max(int(getattr(config, "length_normalizer", 300)), 1)
    bins = list(config.length_bins or [])
    if len(bins) < 2:
        bins = [0.3, 0.6]
    bins = sorted(bins)[:2]
    short_max = max(1, int(round(bins[0] * base)))
    medium_max = max(short_max + 1, int(round(bins[1] * base)))
    short_min = max(20, int(round(short_max * 0.6)))
    medium_min = short_max + 1
    long_min = medium_max + 1
    long_max = max(long_min + 20, int(round(base * 0.95)))
    return {
        "short": (short_min, short_max),
        "medium": (medium_min, medium_max),
        "long": (long_min, long_max),
    }
