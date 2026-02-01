from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List


@dataclass
class LoCoMoRecord:
    question: str
    answer: str
    history: List[str]
    task_type: str | None = None


def load_locomo(path: str) -> List[LoCoMoRecord]:
    # Accept a single file or a directory of JSON/JSONL shards.
    p = Path(path)
    if p.is_dir():
        json_files = list(p.glob("*.json")) + list(p.glob("*.jsonl"))
        if not json_files:
            raise FileNotFoundError(f"No JSON/JSONL files found in {p}")
        records: List[LoCoMoRecord] = []
        for file in json_files:
            records.extend(_load_file(file))
        return records
    return _load_file(p)


def _load_file(path: Path) -> List[LoCoMoRecord]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    return _load_json(path)


def _load_jsonl(path: Path) -> List[LoCoMoRecord]:
    records: List[LoCoMoRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = _normalize_record(obj)
            if rec:
                records.append(rec)
    return records


def _load_json(path: Path) -> List[LoCoMoRecord]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[LoCoMoRecord] = []
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict) and "qa" in obj and "conversation" in obj:
                records.extend(_expand_locomo_bundle(obj))
            else:
                rec = _normalize_record(obj)
                if rec:
                    records.append(rec)
    elif isinstance(data, dict):
        for obj in _iter_dict_records(data):
            rec = _normalize_record(obj)
            if rec:
                records.append(rec)
    return records


def _iter_dict_records(data: dict) -> Iterable[dict]:
    for key in ["records", "data", "examples", "items"]:
        if key in data and isinstance(data[key], list):
            for obj in data[key]:
                yield obj
    if "qa" in data and "conversation" in data:
        for rec in _expand_locomo_bundle(data):
            yield {
                "question": rec.question,
                "answer": rec.answer,
                "history": rec.history,
                "task_type": rec.task_type,
            }


def _expand_locomo_bundle(obj: dict[str, Any]) -> List[LoCoMoRecord]:
    # LoCoMo v2 style: a bundle of sessions + QA pairs.
    qa_list = obj.get("qa", [])
    conversation = obj.get("conversation", {})
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    history = _flatten_conversation(conversation, speaker_a, speaker_b)
    records: List[LoCoMoRecord] = []
    for qa in qa_list:
        question = qa.get("question")
        answer = qa.get("answer")
        task_type = str(qa.get("category")) if qa.get("category") is not None else None
        if question and answer is not None:
            records.append(
                LoCoMoRecord(
                    question=str(question),
                    answer=str(answer),
                    history=history,
                    task_type=str(task_type) if task_type is not None else None,
                )
            )
    return records


def _flatten_conversation(conversation: dict, speaker_a: str | None, speaker_b: str | None) -> List[str]:
    # Sessions are stored as session_1 ... session_n lists of turns.
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
            text = turn.get("text", "")
            role = _map_role(speaker, speaker_a, speaker_b)
            history.append(f"{role}: {text}")
    return history


def _map_role(speaker: str, speaker_a: str | None, speaker_b: str | None) -> str:
    if speaker_a and speaker == speaker_a:
        return "User"
    if speaker_b and speaker == speaker_b:
        return "Assistant"
    return "Speaker"


def _normalize_record(obj: dict[str, Any]) -> LoCoMoRecord | None:
    # Support multiple schema variants used by LoCoMo-like datasets.
    if "qa" in obj and "conversation" in obj:
        # This is a LoCoMo-style bundle: expand in the loader entrypoint.
        return None
    question = obj.get("question") or obj.get("query") or obj.get("prompt")
    answer = obj.get("answer") or obj.get("response") or obj.get("gold")
    history = obj.get("history") or obj.get("dialogue") or obj.get("context") or []
    task_type = obj.get("task_type") or obj.get("task") or obj.get("type")
    if not question or answer is None:
        return None
    history_lines = _normalize_history(history)
    return LoCoMoRecord(
        question=str(question),
        answer=str(answer),
        history=history_lines,
        task_type=str(task_type) if task_type is not None else None,
    )


def _normalize_history(history: Any) -> List[str]:
    # Preserve role labels so the baseline prompt builder can use them directly.
    if isinstance(history, list):
        if history and isinstance(history[0], dict):
            lines: List[str] = []
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                role_cap = "User" if role.lower() == "user" else "Assistant"
                lines.append(f"{role_cap}: {content}")
            return lines
        return [str(h) for h in history]
    if isinstance(history, str):
        return [history]
    return []
