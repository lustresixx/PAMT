from __future__ import annotations

from collections import Counter


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def f1_score(pred: str, gold: str) -> float:
    # Token-level F1 for QA-style responses.
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = pred_counts & gold_counts
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    # Unigram BLEU with brevity penalty (simple BLEU-1 proxy).
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)
    brevity = min(1.0, len(pred_tokens) / max(len(gold_tokens), 1))
    return precision * brevity
