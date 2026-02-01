from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..config import PreferenceConfig
from ..core.types import CategoryWithProb


ToneModel = Callable[[str], CategoryWithProb]
EmotionModel = Callable[[str], CategoryWithProb]
FormalityModel = Callable[[str], float]
OpenIEModel = Callable[[str], List[Tuple[str, str, str]]]


DEFAULT_TONE_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_EMOTION_MODEL_ID = "skep_ernie_1.0_large_ch"
DEFAULT_FORMALITY_MODEL_ID = "s-nlp/roberta-base-formality-ranker"
DEFAULT_DENSITY_MODEL_ID = "wiki80_bert_softmax"
DEFAULT_GLINER_MODEL_ID = "urchade/gliner_small-v2.1"


DEFAULT_GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "time",
    "event",
    "product",
    "work_of_art",
    "law",
    "language",
    "quantity",
    "money",
    "percent",
    "facility",
    "nationality",
    "religion",
    "url",
    "email",
    "phone",
    "role",
    "title",
]


GO_EMOTION_MAP: Dict[str, str] = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "neutral": "neutral",
    "optimism": "joy",
    "pride": "joy",
    "realization": "neutral",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "neutral",
}

TONE_LABEL_ALIASES: Dict[str, str] = {
    "humor": "humor",
    "humorous": "humor",
    "funny": "humor",
    "sarcasm": "humor",
    "irony": "humor",
    "serious": "serious",
    "neutral": "neutral",
    "formal": "formal",
    "casual": "casual",
    "friendly": "friendly",
}

EMOTION_LABEL_ALIASES: Dict[str, str] = {
    "positive": "joy",
    "negative": "sadness",
    "neutral": "neutral",
}


def build_hf_tone_model(
    config: PreferenceConfig,
    model_id: str = DEFAULT_TONE_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
) -> ToneModel:
    # REAL MODEL: HuggingFace classifier for irony/sarcasm detection.
    transformers = _require_optional("transformers")
    classifier = transformers.pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device,
        framework="pt",
    )

    def tone_model(text: str) -> CategoryWithProb:
        scores = _pipeline_scores(classifier, text, max_length)
        return _irony_scores_to_tone(scores, config.tone_labels)

    return tone_model


def build_hf_emotion_model(
    config: PreferenceConfig,
    model_id: str = DEFAULT_EMOTION_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
) -> EmotionModel:
    # REAL MODEL: HuggingFace GoEmotions classifier (maps to base labels).
    transformers = _require_optional("transformers")
    classifier = transformers.pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device,
        framework="pt",
    )

    def emotion_model(text: str) -> CategoryWithProb:
        scores = _pipeline_scores(classifier, text, max_length)
        return _go_emotion_scores_to_distribution(scores, config.emotion_labels)

    return emotion_model


def build_hf_formality_model(
    model_id: str = DEFAULT_FORMALITY_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
) -> FormalityModel:
    # REAL MODEL: HuggingFace formality ranker (continuous 0-1 score).
    transformers = _require_optional("transformers")
    torch = _require_optional("torch")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    torch_device = _resolve_torch_device(torch, device)
    model.to(torch_device)

    def formality_model(text: str) -> float:
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        score = _formality_score_from_logits(logits, model.config, torch)
        return _clamp(score)

    return formality_model


def build_gliner_openie_model(
    model_id: str = DEFAULT_GLINER_MODEL_ID,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    max_entities: int = 64,
    max_text_tokens: int = 384,
    device: Optional[int | str] = None,
) -> OpenIEModel:
    # REAL MODEL: GLiNER entity extractor used as an OpenIE-like triple source.
    gliner = _require_optional("gliner")
    model = gliner.GLiNER.from_pretrained(model_id)
    if device is not None and hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            pass
    label_set = labels or list(DEFAULT_GLINER_LABELS)

    def openie_model(text: str) -> List[Tuple[str, str, str]]:
        text = _truncate_text(text, max_text_tokens)
        entities = model.predict_entities(text, label_set, threshold=threshold)
        triples: List[Tuple[str, str, str]] = []
        for ent in entities[:max_entities]:
            ent_text, ent_label = _extract_entity(ent)
            if not ent_text or not ent_label:
                continue
            triples.append((ent_text, "is", ent_label))
        return triples

    return openie_model


def build_roberta_tone_model(
    config: PreferenceConfig,
    model_id: str = DEFAULT_TONE_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
) -> ToneModel:
    # Preferred: RoBERTa encoder + multi-class head for tone style.
    transformers = _require_optional("transformers")
    classifier = transformers.pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device,
        framework="pt",
    )

    def tone_model(text: str) -> CategoryWithProb:
        scores = _pipeline_scores(classifier, text, max_length)
        probs = _scores_to_distribution(scores, config.tone_labels, TONE_LABEL_ALIASES)
        return _argmax(probs), probs

    return tone_model


def build_skep_emotion_model(
    config: PreferenceConfig,
    model_id: str = DEFAULT_EMOTION_MODEL_ID,
    device: Optional[int | str] = None,
) -> EmotionModel:
    # Preferred: SKEP for emotional tone (sentiment) classification.
    paddlenlp = _require_optional("paddlenlp")
    try:
        from paddlenlp import Taskflow
    except ImportError as exc:
        raise ImportError("Missing paddlenlp Taskflow for SKEP emotion model.") from exc

    try:
        sentiment = Taskflow("sentiment_analysis", model=model_id, device=device)
    except TypeError:
        sentiment = Taskflow("sentiment_analysis", model=model_id)

    def emotion_model(text: str) -> CategoryWithProb:
        raw = sentiment(text)
        scores = _extract_label_scores(raw)
        probs = _scores_to_distribution(scores, config.emotion_labels, EMOTION_LABEL_ALIASES)
        return _argmax(probs), probs

    return emotion_model


def build_opennre_openie_model(
    model_id: str = DEFAULT_DENSITY_MODEL_ID,
    *,
    max_pairs: int = 64,
    max_entities: int = 32,
    max_text_tokens: int = 384,
    spacy_model: str = "en_core_web_sm",
) -> OpenIEModel:
    # Preferred: OpenNRE-based relation extraction for information density.
    opennre = _require_optional("opennre")
    model = opennre.get_model(model_id)

    nlp = None
    try:
        spacy = _require_optional("spacy")
        nlp = spacy.load(spacy_model)
    except Exception:
        nlp = None

    def openie_model(text: str) -> List[Tuple[str, str, str]]:
        text = _truncate_text(text, max_text_tokens)
        entities = _extract_entities(text, nlp, max_entities=max_entities)
        if len(entities) < 2:
            return []
        triples: List[Tuple[str, str, str]] = []
        pair_count = 0
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue
                pair_count += 1
                if pair_count > max_pairs:
                    return triples
                rel = model.infer(
                    {
                        "text": text,
                        "h": {"pos": (head[0], head[1])},
                        "t": {"pos": (tail[0], tail[1])},
                    }
                )
                rel_name = rel[0] if isinstance(rel, (list, tuple)) else rel
                if _is_valid_relation(rel_name):
                    triples.append((head[2], str(rel_name), tail[2]))
        return triples

    return openie_model


def _pipeline_scores(classifier: Any, text: str, max_length: int) -> List[Dict[str, float]]:
    try:
        raw = classifier(text, top_k=None, truncation=True, max_length=max_length)
    except TypeError:
        raw = classifier(text, return_all_scores=True, truncation=True, max_length=max_length)
    if isinstance(raw, dict):
        raw = [raw]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    scores: List[Dict[str, float]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        score = float(item.get("score", 0.0))
        scores.append({"label": label, "score": score})
    return scores


def _scores_to_distribution(
    scores: List[Dict[str, float]],
    labels: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> List[float]:
    if not labels:
        return []
    probs = [0.0] * len(labels)
    for item in scores:
        idx = _map_label_to_index(item.get("label", ""), labels, alias_map)
        if idx is None:
            continue
        probs[idx] += float(item.get("score", 0.0))
    if sum(probs) == 0.0 and scores:
        top = max(scores, key=lambda s: s.get("score", 0.0))
        idx = _map_label_to_index(top.get("label", ""), labels, alias_map)
        if idx is None:
            idx = 0
        probs[idx] = 1.0
    return _normalize_probs(probs)


def _map_label_to_index(
    label: str,
    labels: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> Optional[int]:
    normalized = _normalize_label(label)
    if normalized in labels:
        return labels.index(normalized)
    if alias_map:
        mapped = alias_map.get(normalized)
        if mapped in labels:
            return labels.index(mapped)
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if digits:
        idx = int(digits)
        if 0 <= idx < len(labels):
            return idx
    return None


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace("label_", "")


def _extract_label_scores(raw: Any) -> List[Dict[str, float]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    scores: List[Dict[str, float]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        if "probs" in item and isinstance(item["probs"], dict):
            for label, score in item["probs"].items():
                scores.append({"label": str(label), "score": float(score)})
            continue
        if "scores" in item and isinstance(item["scores"], dict):
            for label, score in item["scores"].items():
                scores.append({"label": str(label), "score": float(score)})
            continue
        label = item.get("label")
        score = item.get("score", 1.0)
        if label is not None:
            scores.append({"label": str(label), "score": float(score)})
    return scores


def _irony_scores_to_tone(scores: List[Dict[str, float]], labels: List[str]) -> CategoryWithProb:
    if not labels:
        return 0, []
    irony_score = 0.0
    non_irony_score = 0.0
    for item in scores:
        label = item["label"].lower().replace("-", "_")
        if "irony" in label or "sarcasm" in label:
            if "non" in label or "not" in label:
                non_irony_score += item["score"]
            else:
                irony_score += item["score"]
    if irony_score == 0.0 and non_irony_score == 0.0 and scores:
        top = max(scores, key=lambda s: s["score"])
        label = top["label"].lower().replace("-", "_")
        if "irony" in label or "sarcasm" in label:
            irony_score = top["score"]
        else:
            non_irony_score = top["score"]
    probs = [0.0] * len(labels)
    if irony_score > 0.0:
        idx = _pick_label_index(labels, ["casual", "friendly", "neutral"])
        probs[idx] += irony_score
    if non_irony_score > 0.0:
        idx = _pick_label_index(labels, ["formal", "neutral"])
        probs[idx] += non_irony_score
    probs = _normalize_probs(probs)
    return _argmax(probs), probs


def _go_emotion_scores_to_distribution(
    scores: List[Dict[str, float]], labels: List[str]
) -> CategoryWithProb:
    if not labels:
        return 0, []
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    probs = [0.0] * len(labels)
    for item in scores:
        label = item["label"].lower().replace("-", "_")
        mapped = GO_EMOTION_MAP.get(label)
        if mapped is None:
            continue
        idx = label_to_idx.get(mapped)
        if idx is None:
            continue
        probs[idx] += item["score"]
    if sum(probs) == 0.0:
        fallback_idx = _pick_label_index(labels, ["neutral"])
        probs[fallback_idx] = 1.0
    probs = _normalize_probs(probs)
    return _argmax(probs), probs


def _formality_score_from_logits(logits: Any, config: Any, torch: Any) -> float:
    if logits.shape[-1] == 1:
        score = torch.sigmoid(logits).squeeze().item()
        return float(score)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    formal_idx = _find_formal_label_idx(getattr(config, "id2label", None))
    if formal_idx is not None and 0 <= formal_idx < probs.shape[-1]:
        return float(probs[formal_idx].item())
    informal_idx = _find_informal_label_idx(getattr(config, "id2label", None))
    if informal_idx is not None and 0 <= informal_idx < probs.shape[-1]:
        return float(1.0 - probs[informal_idx].item())
    # Fallback: assume label 1 is formal if present, else use label 0.
    if probs.shape[-1] > 1:
        return float(probs[1].item())
    return float(probs[0].item())


def _find_formal_label_idx(id2label: Optional[Dict[int, str]]) -> Optional[int]:
    if not id2label:
        return None
    for idx, label in id2label.items():
        text = str(label).lower()
        if "formal" in text and "informal" not in text and "casual" not in text:
            return int(idx)
    return None


def _find_informal_label_idx(id2label: Optional[Dict[int, str]]) -> Optional[int]:
    if not id2label:
        return None
    for idx, label in id2label.items():
        text = str(label).lower()
        if "informal" in text or "casual" in text:
            return int(idx)
    return None


def _is_valid_relation(relation: Any) -> bool:
    if relation is None:
        return False
    rel = str(relation).strip().lower()
    return rel not in {"na", "no_relation", "none", "other", "unknown"}


def _extract_entities(
    text: str,
    nlp: Any,
    *,
    max_entities: int = 32,
) -> List[Tuple[int, int, str]]:
    token_spans = _token_spans(text)
    entities: List[Tuple[int, int, str]] = []
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in getattr(doc, "ents", [])[:max_entities]:
                token_range = _char_span_to_token_range(token_spans, ent.start_char, ent.end_char)
                if token_range is None:
                    continue
                entities.append((token_range[0], token_range[1], ent.text))
            if entities:
                return entities
        except Exception:
            pass
    # Fallback: capitalized tokens as coarse entities.
    tokens = [span[2] for span in token_spans]
    for idx, tok in enumerate(tokens):
        if tok[:1].isupper():
            entities.append((idx, idx + 1, tok))
            if len(entities) >= max_entities:
                break
    return entities


def _token_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    offset = 0
    for token in text.split():
        start = text.find(token, offset)
        if start == -1:
            continue
        end = start + len(token)
        spans.append((start, end, token))
        offset = end
    return spans


def _char_span_to_token_range(
    token_spans: List[Tuple[int, int, str]],
    start_char: int,
    end_char: int,
) -> Tuple[int, int] | None:
    start_idx = None
    end_idx = None
    for i, (start, end, _tok) in enumerate(token_spans):
        if start_idx is None and end > start_char:
            start_idx = i
        if end >= end_char:
            end_idx = i + 1
            break
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return None
    return start_idx, end_idx


def _extract_entity(entity: Any) -> Tuple[str, str]:
    if isinstance(entity, dict):
        return str(entity.get("text", "")), str(entity.get("label", ""))
    if hasattr(entity, "text") and hasattr(entity, "label"):
        return str(getattr(entity, "text", "")), str(getattr(entity, "label", ""))
    if isinstance(entity, (list, tuple)) and len(entity) >= 2:
        return str(entity[0]), str(entity[1])
    return "", ""


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    words = text.split()
    if len(words) > 1:
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return text
    if len(text) > max_tokens:
        return text[:max_tokens]
    return text


def _argmax(values: List[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=values.__getitem__)


def _pick_label_index(labels: List[str], prefer: Iterable[str]) -> int:
    for label in prefer:
        if label in labels:
            return labels.index(label)
    return 0


def _normalize_probs(probs: List[float]) -> List[float]:
    if not probs:
        return probs
    total = sum(probs)
    if total <= 0.0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _resolve_torch_device(torch: Any, device: Optional[int | str]) -> Any:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        if device < 0:
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _require_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. "
            "Install transformers, torch, paddlenlp, opennre, spacy, and gliner to use model-backed extractors."
        ) from exc
