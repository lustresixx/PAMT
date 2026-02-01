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
        "If the user's query specifies content requirements, follow those; otherwise use these preferences."
        f"Tone: {tone_label}, Emotion: {emotion_label}, "
        f"Density: {density_label}, Length: {length_label}, Formality: {formality_label}. "
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
