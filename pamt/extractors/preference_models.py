from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..config import PreferenceConfig
from ..core.types import CategoryWithProb


ToneModel = Callable[[str], CategoryWithProb]
EmotionModel = Callable[[str], CategoryWithProb]
FormalityModel = Callable[[str], float]
OpenIEModel = Callable[[str], List[Tuple[str, str, str]]]

logger = logging.getLogger(__name__)


DEFAULT_TONE_MODEL_ID = "FacebookAI/roberta-large-mnli"
DEFAULT_EMOTION_MODEL_ID = "skep_ernie_1.0_large_ch"
DEFAULT_FORMALITY_MODEL_ID = "s-nlp/roberta-base-formality-ranker"
DEFAULT_DENSITY_MODEL_ID = "wiki80_bert_softmax"


TONE_LABEL_ALIASES: Dict[str, str] = {
    "humor": "humorous",
    "humorous": "humorous",
    "funny": "humorous",
    "sarcasm": "humorous",
    "irony": "humorous",
    "serious": "serious",
    "gentle": "gentle",
    "neutral": "neutral",
    "formal": "serious",
    "casual": "humorous",
    "friendly": "gentle",
}

EMOTION_LABEL_ALIASES: Dict[str, str] = {
    "positive": "joy",
    "negative": "sadness",
    "neutral": "neutral",
}


def build_roberta_tone_model(
    config: PreferenceConfig,
    model_id: str = DEFAULT_TONE_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
    cache_dir: str | None = None,
) -> ToneModel:
    """RoBERTa encoder + classification head for tone style (paper model)."""
    transformers = _require_optional("transformers")
    torch = _require_optional("torch")
    token = _get_hf_token()
    cache_dir = _get_hf_cache_dir(cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    if device is None:
        device_id = 0 if torch.cuda.is_available() else -1
    elif isinstance(device, int):
        device_id = device if device >= 0 else -1
    else:
        device_str = str(device).lower()
        if device_str.startswith("cuda") or device_str.startswith("gpu"):
            device_id = 0
        else:
            device_id = -1
    classifier = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
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
    """SKEP emotional tone classifier (paper model)."""
    _require_optional("paddlenlp")
    try:
        paddle = _require_optional("paddle")
        tokenizer, model = _load_skep_taskflow_model(model_id, device, paddle)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize SKEP emotion model: {exc}") from exc

    def emotion_model(text: str) -> CategoryWithProb:
        inputs = _skep_tokenize(tokenizer, text)
        inputs = {k: paddle.to_tensor([v]) for k, v in inputs.items() if v is not None}
        with paddle.no_grad():
            _idx, probs = model(**inputs)
        probs_list = probs.numpy()[0].tolist()
        scores = [
            {"label": "negative", "score": float(probs_list[0])},
            {"label": "positive", "score": float(probs_list[1])},
        ]
        probs = _scores_to_distribution(scores, config.emotion_labels, EMOTION_LABEL_ALIASES)
        return _argmax(probs), probs

    return emotion_model


def build_hf_formality_model(
    model_id: str = DEFAULT_FORMALITY_MODEL_ID,
    device: Optional[int | str] = None,
    max_length: int = 256,
    cache_dir: str | None = None,
) -> FormalityModel:
    """Formality classifier (paper model)."""
    transformers = _require_optional("transformers")
    torch = _require_optional("torch")
    token = _get_hf_token()
    cache_dir = _get_hf_cache_dir(cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
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


def build_opennre_openie_model(
    model_id: str = DEFAULT_DENSITY_MODEL_ID,
    *,
    device: Optional[int | str] = None,
    max_pairs: int = 64,
    max_entities: int = 32,
    max_text_tokens: int = 384,
    spacy_model: str = "en_core_web_sm",
) -> OpenIEModel:
    """OpenNRE-based relation extraction for information density (paper model)."""
    _ensure_home_env()
    _ensure_opennre_assets(model_id)
    opennre = _require_optional("opennre")
    model = _load_opennre_model(opennre, model_id)
    torch = _require_optional("torch")
    torch_device = _resolve_torch_device(torch, device)
    if hasattr(model, "to"):
        model.to(torch_device)
    if hasattr(model, "device"):
        try:
            model.device = torch_device
        except Exception:
            pass
    if hasattr(model, "eval"):
        model.eval()

    nlp = _load_spacy_model(spacy_model)

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
    if nlp is None:
        raise RuntimeError("spaCy model is required for OpenNRE entity extraction.")
    token_spans = _token_spans(text)
    entities: List[Tuple[int, int, str]] = []
    doc = nlp(text)
    for ent in getattr(doc, "ents", [])[:max_entities]:
        token_range = _char_span_to_token_range(token_spans, ent.start_char, ent.end_char)
        if token_range is None:
            continue
        entities.append((token_range[0], token_range[1], ent.text))
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


def _ensure_home_env() -> None:
    if os.environ.get("HOME"):
        return
    home = os.environ.get("USERPROFILE")
    if not home:
        drive = os.environ.get("HOMEDRIVE")
        path = os.environ.get("HOMEPATH")
        if drive and path:
            home = drive + path
    if not home:
        try:
            from pathlib import Path

            home = str(Path.home())
        except Exception:
            home = None
    if home:
        os.environ["HOME"] = home


def _ensure_paddlenlp_static_model(model_id: str) -> None:
    try:
        from paddlenlp.utils.env import PPNLP_HOME
    except Exception:
        return
    task_path = os.path.join(PPNLP_HOME, "taskflow", "sentiment_analysis", model_id)
    static_dir = os.path.join(task_path, "static")
    pdiparams = os.path.join(static_dir, "inference.pdiparams")
    pdmodel = os.path.join(static_dir, "inference.pdmodel")
    if os.path.exists(pdiparams) and not os.path.exists(pdmodel):
        try:
            os.remove(pdiparams)
        except OSError:
            pass


def _load_skep_taskflow_model(model_id: str, device: Optional[int | str], paddle: Any):
    from paddlenlp.taskflow.sentiment_analysis import SkepTask
    from paddlenlp.taskflow.utils import download_file
    from paddlenlp.taskflow.models.sentiment_analysis_model import SkepSequenceModel
    from paddlenlp.transformers import SkepConfig, SkepTokenizer
    from paddlenlp.utils.env import PPNLP_HOME

    _set_paddle_device(device, paddle)

    task_path = os.path.join(PPNLP_HOME, "taskflow", "sentiment_analysis", model_id)
    os.makedirs(task_path, exist_ok=True)

    for file_id, file_name in SkepTask.resource_files_names.items():
        if model_id not in SkepTask.resource_files_urls:
            continue
        if file_id not in SkepTask.resource_files_urls[model_id]:
            continue
        url, md5 = SkepTask.resource_files_urls[model_id][file_id]
        target = os.path.join(task_path, file_name)
        if not os.path.exists(target):
            download_file(task_path, file_name, url, md5)

    state_path = os.path.join(task_path, "model_state.pdparams")
    state = paddle.load(state_path)
    num_labels = _infer_num_labels(state)

    config = SkepConfig.from_pretrained(model_id)
    config.num_labels = num_labels

    model = SkepSequenceModel(config)
    model.set_state_dict(state)
    model.eval()

    tokenizer = SkepTokenizer.from_pretrained(model_id)
    return tokenizer, model


def _infer_num_labels(state: Dict[str, Any]) -> int:
    weight = state.get("classifier.weight")
    if weight is None:
        return 2
    try:
        shape = list(weight.shape)
        if len(shape) == 2:
            if shape[0] <= 10:
                return int(shape[0])
            if shape[1] <= 10:
                return int(shape[1])
        return int(shape[-1])
    except Exception:
        return 2


def _set_paddle_device(device: Optional[int | str], paddle: Any) -> None:
    if device is None:
        if getattr(paddle, "is_compiled_with_cuda", lambda: False)():
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
        return
    if isinstance(device, int):
        if device >= 0 and getattr(paddle, "is_compiled_with_cuda", lambda: False)():
            paddle.set_device(f"gpu:{device}")
        else:
            paddle.set_device("cpu")
        return
    device_str = str(device).lower()
    if device_str.startswith("cuda"):
        device_str = device_str.replace("cuda", "gpu", 1)
    if device_str.startswith("gpu") or device_str == "cpu":
        paddle.set_device(device_str)
    else:
        paddle.set_device("cpu")


def _skep_tokenize(tokenizer: Any, text: str) -> Dict[str, Any]:
    try:
        return tokenizer(text, max_length=256, truncation=True)
    except TypeError:
        try:
            return tokenizer(text, max_seq_len=256)
        except TypeError:
            return tokenizer(text)


def _load_spacy_model(model_name: str):
    spacy = _require_optional("spacy")
    try:
        return spacy.load(model_name)
    except Exception as exc:
        try:
            from spacy.cli import download as spacy_download

            spacy_download(model_name)
            return spacy.load(model_name)
        except Exception as download_exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is required for OpenNRE; "
                f"run: python -m spacy download {model_name}"
            ) from download_exc


def _ensure_opennre_assets(model_id: str) -> None:
    _ensure_home_env()
    root_path = os.path.join(os.environ.get("HOME") or "", ".opennre")
    if not root_path:
        raise RuntimeError("OpenNRE HOME path not set.")
    _ensure_dir(os.path.join(root_path, "benchmark", "wiki80"))
    _ensure_dir(os.path.join(root_path, "pretrain", "nre"))
    _ensure_dir(os.path.join(root_path, "pretrain", "bert-base-uncased"))
    _ensure_dir(os.path.join(root_path, "pretrain", "glove"))

    base = "https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre"
    _download_if_missing(
        f"{base}/benchmark/wiki80/wiki80_rel2id.json",
        os.path.join(root_path, "benchmark", "wiki80", "wiki80_rel2id.json"),
    )

    if model_id in {"wiki80_bert_softmax", "wiki80_bertentity_softmax"}:
        _download_if_missing(
            f"{base}/pretrain/nre/{model_id}.pth.tar",
            os.path.join(root_path, "pretrain", "nre", f"{model_id}.pth.tar"),
        )
        _download_if_missing(
            f"{base}/pretrain/bert-base-uncased/config.json",
            os.path.join(root_path, "pretrain", "bert-base-uncased", "config.json"),
        )
        _download_if_missing(
            f"{base}/pretrain/bert-base-uncased/pytorch_model.bin",
            os.path.join(root_path, "pretrain", "bert-base-uncased", "pytorch_model.bin"),
        )
        _download_if_missing(
            f"{base}/pretrain/bert-base-uncased/vocab.txt",
            os.path.join(root_path, "pretrain", "bert-base-uncased", "vocab.txt"),
        )
    elif model_id == "wiki80_cnn_softmax":
        _download_if_missing(
            f"{base}/pretrain/nre/{model_id}.pth.tar",
            os.path.join(root_path, "pretrain", "nre", f"{model_id}.pth.tar"),
        )
        _download_if_missing(
            f"{base}/pretrain/glove/glove.6B.50d_word2id.json",
            os.path.join(root_path, "pretrain", "glove", "glove.6B.50d_word2id.json"),
        )
        _download_if_missing(
            f"{base}/pretrain/glove/glove.6B.50d_mat.npy",
            os.path.join(root_path, "pretrain", "glove", "glove.6B.50d_mat.npy"),
        )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download_if_missing(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    _ensure_dir(os.path.dirname(dest))
    logger.info("Downloading OpenNRE asset: %s", url)
    urllib.request.urlretrieve(url, dest)


def _load_opennre_model(opennre: Any, model_id: str) -> Any:
    try:
        return opennre.get_model(model_id)
    except RuntimeError as exc:
        message = str(exc)
        if "Unexpected key(s) in state_dict" not in message or "position_ids" not in message:
            raise
        logger.warning("OpenNRE checkpoint mismatch detected; reloading with strict=False.")
    except Exception:
        logger.warning("OpenNRE get_model failed; attempting manual load.", exc_info=True)

    torch = _require_optional("torch")
    root_path = os.path.join(os.environ.get("HOME") or "", ".opennre")
    if not root_path:
        raise RuntimeError("OpenNRE HOME path not set.")

    ckpt = os.path.join(root_path, "pretrain", "nre", f"{model_id}.pth.tar")
    state = torch.load(ckpt, map_location="cpu")
    state_dict = state.get("state_dict") if isinstance(state, dict) else state
    if not isinstance(state_dict, dict):
        raise RuntimeError("OpenNRE checkpoint is missing a state_dict.")

    if model_id in {"wiki80_bert_softmax", "wiki80_bertentity_softmax"}:
        rel2id_path = os.path.join(root_path, "benchmark", "wiki80", "wiki80_rel2id.json")
        with open(rel2id_path, encoding="utf-8") as handle:
            rel2id = json.load(handle)
        pretrain_path = os.path.join(root_path, "pretrain", "bert-base-uncased")
        if "entity" in model_id:
            sentence_encoder = opennre.encoder.BERTEntityEncoder(
                max_length=80,
                pretrain_path=pretrain_path,
            )
        else:
            sentence_encoder = opennre.encoder.BERTEncoder(
                max_length=80,
                pretrain_path=pretrain_path,
            )
        model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        _load_state_dict_relaxed(model, state_dict)
        return model

    if model_id in {"tacred_bert_softmax", "tacred_bertentity_softmax"}:
        rel2id_path = os.path.join(root_path, "benchmark", "tacred", "tacred_rel2id.json")
        with open(rel2id_path, encoding="utf-8") as handle:
            rel2id = json.load(handle)
        pretrain_path = os.path.join(root_path, "pretrain", "bert-base-uncased")
        if "entity" in model_id:
            sentence_encoder = opennre.encoder.BERTEntityEncoder(
                max_length=80,
                pretrain_path=pretrain_path,
            )
        else:
            sentence_encoder = opennre.encoder.BERTEncoder(
                max_length=80,
                pretrain_path=pretrain_path,
            )
        model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        _load_state_dict_relaxed(model, state_dict)
        return model

    return opennre.get_model(model_id)


def _load_state_dict_relaxed(model: Any, state_dict: Dict[str, Any]) -> None:
    for key in [k for k in state_dict.keys() if k.endswith("position_ids")]:
        state_dict.pop(key, None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("OpenNRE state_dict loaded with missing=%s unexpected=%s", missing, unexpected)


def _require_optional(package: str) -> Any:
    try:
        if package == "transformers":
            return _import_transformers_no_torchvision()
        if package == "opennre":
            _import_transformers_no_torchvision()
            return __import__(package)
        return __import__(package)
    except Exception as exc:
        raise ImportError(f"Missing dependency: {package}") from exc


def _import_transformers_no_torchvision() -> Any:
    import importlib.util
    import sys

    _ensure_hf_endpoint()

    if "transformers" in sys.modules:
        return __import__("transformers")

    # Block torchvision before torch is imported to avoid conflicts
    original_find_spec = importlib.util.find_spec

    def _patched_find_spec(name: str, package: str | None = None):
        if name == "torchvision" or (isinstance(name, str) and name.startswith("torchvision.")):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = _patched_find_spec

    try:
        return __import__("transformers")
    finally:
        importlib.util.find_spec = original_find_spec


def _get_hf_token() -> str | bool | None:
    token = (
        os.environ.get("PAMT_HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        return False
    return token


def _get_hf_cache_dir(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("HF_HOME"):
        return None
    return os.environ.get("PAMT_HF_CACHE_DIR")


def _ensure_hf_endpoint() -> None:
    endpoint = os.environ.get("PAMT_HF_ENDPOINT")
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
