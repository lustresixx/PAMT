from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter

import regex
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()


def _normalize_answer(text: str) -> str:
    # Match LoCoMo-style normalization (lower, strip punctuation, drop articles).
    text = text.replace(",", "")

    def remove_articles(value: str) -> str:
        return regex.sub(r"\b(a|an|the|and)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    normalized = unicodedata.normalize("NFD", text)
    return white_space_fix(remove_articles(remove_punc(lower(normalized))))


def _tokenize(text: str) -> list[str]:
    return [_STEMMER.stem(token) for token in _normalize_answer(text).split() if token]


def f1_score(pred: str, gold: str) -> float:
    # Token-level F1 for QA-style responses with stemming + normalization.
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    # Unigram BLEU with brevity penalty using normalized tokens.
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = overlap / len(pred_tokens)
    ref_len = len(gold_tokens)
    cand_len = len(pred_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / cand_len))
    return precision * brevity_penalty
