from __future__ import annotations

import argparse
import json
import os
import random
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib import error as urlerror
from urllib import request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pamt.config import EmbeddingConfig, PreferenceConfig, UpdateConfig
from pamt.core.prompting import format_preference
from pamt.core.types import PreferenceFusion
from pamt.core.update import PAMTState, PAMTUpdater
from pamt.embeddings.models import HFLocalEmbeddings
from pamt.extractors.preference_extractor import ExtractionState, ModelPreferenceExtractor
from pamt.metrics import bleu1_score, f1_score


TASK_CATEGORY_MAP = {
    "single-hop": {4},
    "multi-hop": {1},
    "temporal": {2},
}


@dataclass
class QAItem:
    sample_id: str
    question: str
    answer: str
    category: int
    conversation: dict


@dataclass
class MemoryIndex:
    notes: List[str]
    embeddings: List[List[float]]

    def retrieve(self, query: str, embedder: HFLocalEmbeddings, top_k: int) -> List[str]:
        if not self.notes:
            return []
        query_emb = embedder.embed(query)
        scored = []
        for note, emb in zip(self.notes, self.embeddings):
            score = _cosine_similarity(query_emb, emb)
            scored.append((score, note))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [note for _score, note in scored[:top_k]]


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model_name: str,
        api_base_url: str,
        api_key: str | None = None,
        request_timeout: int = 600,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key or ""
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, *, retries: int = 3, retry_delay: float = 1.0) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.api_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc = None
        for _idx in range(max(retries, 1)):
            req = request.Request(url, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.request_timeout) as resp:
                    body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["choices"][0]["message"]["content"]
            except (urlerror.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_exc = exc
                time.sleep(retry_delay)
        if last_exc:
            raise RuntimeError(f"LLM request failed: {last_exc}") from last_exc
        raise RuntimeError("LLM request failed with unknown error.")


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_locomo_raw(path: str) -> List[dict]:
    payload = _load_json(Path(path))
    if isinstance(payload, list):
        return payload
    raise ValueError("LoCoMo payload must be a JSON list.")


def iter_qa_items(conversations: Iterable[dict], categories: set[int]) -> List[QAItem]:
    items: List[QAItem] = []
    for conv in conversations:
        sample_id = str(conv.get("sample_id", "sample"))
        for qa in conv.get("qa", []):
            category = qa.get("category")
            if category not in categories:
                continue
            question = qa.get("question")
            answer = qa.get("answer")
            if not question or answer is None:
                continue
            items.append(
                QAItem(
                    sample_id=sample_id,
                    question=str(question),
                    answer=_normalize_answer(answer),
                    category=int(category),
                    conversation=conv,
                )
            )
    return items


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return ", ".join(str(x) for x in answer)
    return str(answer)


def _flatten_conversation(conversation: dict) -> List[str]:
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    sessions = []
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            try:
                idx = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            sessions.append((idx, value))
    sessions.sort(key=lambda x: x[0])
    history: List[str] = []
    for _idx, turns in sessions:
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            role = _map_role(speaker, speaker_a, speaker_b)
            history.append(f"{role}: {text}")
    return history


def _map_role(speaker: str, speaker_a: str | None, speaker_b: str | None) -> str:
    if speaker_a and speaker == speaker_a:
        return "User"
    if speaker_b and speaker == speaker_b:
        return "Assistant"
    return "Speaker"


def _extract_turn_pairs(history: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    last_user: str | None = None
    for line in history:
        if line.startswith("User:"):
            last_user = line.split("User:", 1)[1].strip()
        elif line.startswith("Assistant:"):
            if last_user is None:
                continue
            assistant = line.split("Assistant:", 1)[1].strip()
            pairs.append((last_user, assistant))
            last_user = None
    return pairs


def build_amem_notes(
    conv: dict,
    *,
    use_session_summary: bool = True,
    use_observation: bool = True,
    use_event_summary: bool = True,
) -> List[str]:
    notes: List[str] = []
    if use_session_summary:
        session_summary = conv.get("session_summary", {})
        if isinstance(session_summary, dict):
            for text in session_summary.values():
                _add_note(notes, text)

    if use_observation:
        observation = conv.get("observation", {})
        if isinstance(observation, dict):
            for speaker_map in observation.values():
                if not isinstance(speaker_map, dict):
                    continue
                for items in speaker_map.values():
                    if isinstance(items, list):
                        for entry in items:
                            if isinstance(entry, (list, tuple)) and entry:
                                _add_note(notes, entry[0])
                            else:
                                _add_note(notes, entry)

    if use_event_summary:
        event_summary = conv.get("event_summary", {})
        if isinstance(event_summary, dict):
            for payload in event_summary.values():
                if not isinstance(payload, dict):
                    continue
                date = payload.get("date")
                for speaker, events in payload.items():
                    if speaker == "date":
                        continue
                    if isinstance(events, list):
                        for ev in events:
                            text = f"{speaker}: {ev}"
                            if date:
                                text = f"{text} (date: {date})"
                            _add_note(notes, text)

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        deduped.append(note)
    return deduped


def _add_note(notes: List[str], text: Any) -> None:
    if text is None:
        return
    note = str(text).strip()
    if not note:
        return
    notes.append(note)


def build_memory_index(
    conv: dict,
    embedder: HFLocalEmbeddings,
    *,
    use_session_summary: bool = True,
    use_observation: bool = True,
    use_event_summary: bool = True,
    cache_dir: Path | None = None,
) -> MemoryIndex:
    cache_path = None
    sample_id = str(conv.get("sample_id", "sample"))
    if cache_dir:
        suffix = f"s{int(use_session_summary)}o{int(use_observation)}e{int(use_event_summary)}"
        model_tag = _slugify(embedder.config.model_name)
        cache_path = cache_dir / f"{sample_id}_amem_notes_{suffix}_{model_tag}.json"
        if cache_path.is_file():
            cached = _load_json(cache_path)
            notes = cached.get("notes", [])
            embeddings = cached.get("embeddings", [])
            if isinstance(notes, list) and isinstance(embeddings, list) and len(notes) == len(embeddings):
                return MemoryIndex(notes=notes, embeddings=embeddings)

    notes = build_amem_notes(
        conv,
        use_session_summary=use_session_summary,
        use_observation=use_observation,
        use_event_summary=use_event_summary,
    )
    embeddings = [embedder.embed(note) for note in notes]
    index = MemoryIndex(notes=notes, embeddings=embeddings)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"notes": notes, "embeddings": embeddings}, ensure_ascii=True),
            encoding="utf-8",
        )
    return index


def build_preference_fusion(
    conv: dict,
    extractor: ModelPreferenceExtractor,
    updater: PAMTUpdater,
    *,
    max_turns: int = 0,
    stride: int = 1,
    min_turns: int = 0,
    cache_dir: Path | None = None,
) -> PreferenceFusion | None:
    cache_path = None
    sample_id = str(conv.get("sample_id", "sample"))
    if cache_dir:
        turn_tag = f"t{max_turns}" if max_turns else "tall"
        cache_path = cache_dir / f"{sample_id}_pamu_pref_{turn_tag}_s{stride}.json"
        if cache_path.is_file():
            cached = _load_json(cache_path)
            return _fusion_from_payload(cached)

    history = _flatten_conversation(conv.get("conversation", {}))
    pairs = _extract_turn_pairs(history)
    if stride > 1:
        pairs = pairs[::stride]
    if max_turns and max_turns > 0:
        pairs = pairs[-max_turns:]
    if min_turns and len(pairs) < min_turns:
        return None
    if not pairs:
        return None
    state = PAMTState()
    ext_state = ExtractionState()
    fusion = None
    for user_text, assistant_text in pairs:
        pref = extractor.extract(user_text, assistant_text, ext_state)
        fusion, _ = updater.update(state, pref)

    if cache_path and fusion is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(_fusion_payload(fusion), ensure_ascii=True), encoding="utf-8")
    return fusion


def _fusion_payload(fusion: PreferenceFusion) -> Dict[str, Any]:
    return {
        "tone": fusion.tone,
        "emotion": fusion.emotion,
        "length": fusion.length,
        "density": fusion.density,
        "formality": fusion.formality,
    }


def _fusion_from_payload(payload: Any) -> PreferenceFusion | None:
    if not isinstance(payload, dict):
        return None
    try:
        return PreferenceFusion(
            tone=tuple(payload["tone"]),
            emotion=tuple(payload["emotion"]),
            length=float(payload["length"]),
            density=float(payload["density"]),
            formality=float(payload["formality"]),
        )
    except Exception:
        return None


def ensure_ollama_model(model_name: str) -> None:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return
    if result.returncode != 0:
        return
    if model_name in result.stdout:
        return
    subprocess.run(["ollama", "pull", model_name], check=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark: AMEM vs AMEM+PAMU (Qwen2.5-3B).")
    parser.add_argument("--data", default="data/locomo10.json", help="Path to LoCoMo JSON.")
    parser.add_argument("--task", default="single-hop", choices=sorted(TASK_CATEGORY_MAP.keys()))
    parser.add_argument("--model-name", default="qwen2.5:3b")
    parser.add_argument("--api-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default=os.environ.get("PAMT_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pamu-top-k", type=int, default=0)
    parser.add_argument("--notes-session-summary", action="store_true")
    parser.add_argument("--notes-observation", action="store_true")
    parser.add_argument("--notes-event-summary", action="store_true")
    parser.add_argument("--pref-max-turns", type=int, default=0)
    parser.add_argument("--pref-stride", type=int, default=1)
    parser.add_argument("--pref-min-turns", type=int, default=6)
    parser.add_argument("--pref-window-size", type=int, default=0)
    parser.add_argument("--pref-ema-decay", type=float, default=-1.0)
    parser.add_argument("--pref-fuse-weight", type=float, default=-1.0)
    parser.add_argument("--pref-change-threshold", type=float, default=-1.0)
    parser.add_argument("--pref-length-history", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache-dir", default="data/benchmark_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-pamu", action="store_true")
    parser.add_argument("--ensure-ollama", action="store_true")
    parser.add_argument(
        "--results-path",
        default="runs/benchmark_results.jsonl",
        help="Append a JSONL record of each run to this path (empty to disable).",
    )
    return parser.parse_args()


def _maybe_write_results(
    args: argparse.Namespace,
    results: Dict[str, Dict[str, float]],
    elapsed_seconds: float,
) -> None:
    if not args.results_path:
        return
    path = Path(args.results_path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "task": args.task,
        "data": str(args.data),
        "model_name": args.model_name,
        "api_base_url": args.api_base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "pamu_top_k": args.pamu_top_k,
        "notes_session_summary": bool(args.notes_session_summary),
        "notes_observation": bool(args.notes_observation),
        "notes_event_summary": bool(args.notes_event_summary),
        "pref_max_turns": args.pref_max_turns,
        "pref_stride": args.pref_stride,
        "pref_min_turns": args.pref_min_turns,
        "pref_window_size": args.pref_window_size,
        "pref_ema_decay": args.pref_ema_decay,
        "pref_fuse_weight": args.pref_fuse_weight,
        "pref_change_threshold": args.pref_change_threshold,
        "pref_length_history": args.pref_length_history,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
        "results": results,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"[run-log] appended results to {path}")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING)
    started_at = time.time()
    if not args.run_baseline and not args.run_pamu:
        args.run_baseline = True
        args.run_pamu = True

    if args.ensure_ollama and "localhost:11434" in args.api_base_url:
        ensure_ollama_model(args.model_name)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    cache_dir = None if args.no_cache else Path(args.cache_dir)
    if cache_dir and not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    embed_config = EmbeddingConfig()
    embedder = HFLocalEmbeddings(embed_config)

    pref_config = PreferenceConfig()
    if args.pref_length_history and args.pref_length_history > 0:
        pref_config.length_history = args.pref_length_history
    update_config = UpdateConfig()
    if args.pref_window_size and args.pref_window_size > 0:
        update_config.window_size = args.pref_window_size
    if args.pref_ema_decay >= 0.0:
        update_config.ema_decay = args.pref_ema_decay
    if args.pref_fuse_weight >= 0.0:
        update_config.fuse_weight = args.pref_fuse_weight
    if args.pref_change_threshold >= 0.0:
        update_config.change_threshold = args.pref_change_threshold
    updater = PAMTUpdater(update_config)
    extractor = ModelPreferenceExtractor.from_preferred_models(pref_config)

    client = OpenAIChatClient(
        model_name=_normalize_model_name(args.model_name),
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    conversations = load_locomo_raw(str(data_path))
    categories = TASK_CATEGORY_MAP[args.task]
    qa_items = iter_qa_items(conversations, categories)

    random.seed(args.seed)
    random.shuffle(qa_items)
    if args.max_samples and args.max_samples > 0:
        qa_items = qa_items[: args.max_samples]

    indexes: Dict[str, MemoryIndex] = {}
    prefs: Dict[str, PreferenceFusion | None] = {}

    results = {}
    note_sources = _note_source_flags(args)

    if args.prepare_cache:
        _prepare_cache(
            qa_items,
            embedder,
            indexes,
            extractor,
            updater,
            prefs,
            cache_dir,
            pref_max_turns=args.pref_max_turns,
            pref_stride=args.pref_stride,
            pref_min_turns=args.pref_min_turns,
            note_sources=note_sources,
        )
        print("Cache prepared.")
        return

    if args.run_baseline:
        results["baseline"] = _evaluate(
            qa_items,
            client,
            embedder,
            indexes,
            prefs,
            pref_config,
            extractor,
            updater,
            top_k=args.top_k,
            pamu_top_k=args.pamu_top_k,
            cache_dir=cache_dir,
            use_pamu=False,
            pref_max_turns=args.pref_max_turns,
            pref_stride=args.pref_stride,
            pref_min_turns=args.pref_min_turns,
            note_sources=note_sources,
        )
    if args.run_pamu:
        results["pamu"] = _evaluate(
            qa_items,
            client,
            embedder,
            indexes,
            prefs,
            pref_config,
            extractor,
            updater,
            top_k=args.top_k,
            pamu_top_k=args.pamu_top_k,
            cache_dir=cache_dir,
            use_pamu=True,
            pref_max_turns=args.pref_max_turns,
            pref_stride=args.pref_stride,
            pref_min_turns=args.pref_min_turns,
            note_sources=note_sources,
        )

    for name, metrics in results.items():
        print(f"[{name}] samples={metrics['count']} f1={metrics['f1']:.2f} bleu1={metrics['bleu1']:.2f}")
    _maybe_write_results(args, results, time.time() - started_at)


def _normalize_model_name(model_name: str) -> str:
    if "qwen2.5/" in model_name:
        return model_name.replace("qwen2.5/", "qwen2.5:", 1)
    return model_name


def _evaluate(
    qa_items: List[QAItem],
    client: OpenAIChatClient,
    embedder: HFLocalEmbeddings,
    indexes: Dict[str, MemoryIndex],
    prefs: Dict[str, PreferenceFusion | None],
    pref_config: PreferenceConfig,
    extractor: ModelPreferenceExtractor,
    updater: PAMTUpdater,
    *,
    top_k: int,
    pamu_top_k: int,
    cache_dir: Path | None,
    use_pamu: bool,
    pref_max_turns: int,
    pref_stride: int,
    pref_min_turns: int,
    note_sources: Tuple[bool, bool, bool],
) -> Dict[str, float]:
    f1_total = 0.0
    bleu_total = 0.0
    count = 0
    for item in qa_items:
        index = _get_index(item, embedder, indexes, cache_dir, note_sources=note_sources)
        pref = None
        if use_pamu:
            pref = _get_pref(
                item,
                extractor,
                updater,
                prefs,
                cache_dir,
                pref_max_turns=pref_max_turns,
                pref_stride=pref_stride,
                pref_min_turns=pref_min_turns,
            )

        effective_top_k = pamu_top_k if use_pamu and pamu_top_k > 0 else top_k
        notes = index.retrieve(item.question, embedder, effective_top_k)
        pref_desc = format_preference(pref, pref_config) if pref is not None else None
        prompt = _build_amem_prompt(notes, item.question, pref_desc=pref_desc)

        prediction = client.generate(prompt)
        f1 = f1_score(prediction, item.answer)
        bleu = bleu1_score(prediction, item.answer)
        f1_total += f1
        bleu_total += bleu
        count += 1

    return {
        "count": count,
        "f1": (f1_total / count * 100.0) if count else 0.0,
        "bleu1": (bleu_total / count * 100.0) if count else 0.0,
    }


def _get_index(
    item: QAItem,
    embedder: HFLocalEmbeddings,
    indexes: Dict[str, MemoryIndex],
    cache_dir: Path | None,
    *,
    note_sources: Tuple[bool, bool, bool],
) -> MemoryIndex:
    key = item.sample_id
    if key not in indexes:
        use_session_summary, use_observation, use_event_summary = note_sources
        indexes[key] = build_memory_index(
            item.conversation,
            embedder,
            use_session_summary=use_session_summary,
            use_observation=use_observation,
            use_event_summary=use_event_summary,
            cache_dir=cache_dir,
        )
    return indexes[key]


def _get_pref(
    item: QAItem,
    extractor: ModelPreferenceExtractor,
    updater: PAMTUpdater,
    prefs: Dict[str, PreferenceFusion | None],
    cache_dir: Path | None,
    *,
    pref_max_turns: int,
    pref_stride: int,
    pref_min_turns: int,
) -> PreferenceFusion | None:
    key = item.sample_id
    if key not in prefs:
        prefs[key] = build_preference_fusion(
            item.conversation,
            extractor,
            updater,
            max_turns=pref_max_turns,
            stride=max(pref_stride, 1),
            min_turns=pref_min_turns,
            cache_dir=cache_dir,
        )
    return prefs[key]


def _prepare_cache(
    qa_items: List[QAItem],
    embedder: HFLocalEmbeddings,
    indexes: Dict[str, MemoryIndex],
    extractor: ModelPreferenceExtractor,
    updater: PAMTUpdater,
    prefs: Dict[str, PreferenceFusion | None],
    cache_dir: Path | None,
    *,
    pref_max_turns: int,
    pref_stride: int,
    pref_min_turns: int,
    note_sources: Tuple[bool, bool, bool],
) -> None:
    for item in qa_items:
        _get_index(item, embedder, indexes, cache_dir, note_sources=note_sources)
        _get_pref(
            item,
            extractor,
            updater,
            prefs,
            cache_dir,
            pref_max_turns=pref_max_turns,
            pref_stride=pref_stride,
            pref_min_turns=pref_min_turns,
        )


def _build_amem_prompt(
    notes: List[str], question: str, *, pref_desc: str | None = None
) -> str:
    memory = "\n".join(f"{idx + 1}. {note}" for idx, note in enumerate(notes)) or "None"
    pref_line = f"Respond in style: {pref_desc}\n" if pref_desc else ""
    return (
        "You are a helpful assistant.\n"
        "Based on the memory notes, answer the question in the form of a short phrase.\n"
        "Answer with exact words from the memory notes whenever possible.\n"
        "If the answer is not in the notes, reply with 'No information available'.\n"
        f"{pref_line}"
        f"Memory notes:\n{memory}\n\nQuestion: {question}\nShort answer:"
    )


def _note_source_flags(args: argparse.Namespace) -> Tuple[bool, bool, bool]:
    # Default to all sources when no flags are provided.
    if not args.notes_session_summary and not args.notes_observation and not args.notes_event_summary:
        return True, True, True
    return args.notes_session_summary, args.notes_observation, args.notes_event_summary


if __name__ == "__main__":
    main()
