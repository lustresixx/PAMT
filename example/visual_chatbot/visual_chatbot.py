from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from langfuse import Langfuse
    from langfuse import observe
    from langfuse.api.client import FernLangfuse
    from langfuse.api.resources.ingestion.types import (
        CreateGenerationBody,
        CreateSpanBody,
        IngestionEvent_GenerationCreate,
        IngestionEvent_GenerationUpdate,
        IngestionEvent_SpanCreate,
        IngestionEvent_SpanUpdate,
        IngestionEvent_TraceCreate,
        TraceBody,
        UpdateGenerationBody,
        UpdateSpanBody,
    )
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    langfuse_context = None
    observe = None
    FernLangfuse = None

from pamt.config import EmbeddingConfig, ModelConfig, PreferenceConfig, UpdateConfig
from pamt.core.memory_tree import HierarchicalMemoryTree, RetrievalConfig
from pamt.core.prompting import build_prompt
from pamt.extractors.preference_extractor import (
    ExtractionState,
    PreferenceExtractor,
    build_preference_extractor,
)
from pamt.embeddings.models import DeepSeekEmbeddings, EmbeddingClient, HFLocalEmbeddings, OllamaEmbeddings
from pamt.llms.models import DeepSeekLLM, LLM, OllamaLLM
from pamt.logging_utils import setup_logging

# Global langfuse client (initialized in main if enabled)
_langfuse_client: Optional["Langfuse"] = None
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    if not value:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _prompt_preview(prompt: str, max_len: int = 2000) -> str:
    if max_len <= 0 or len(prompt) <= max_len:
        return prompt
    truncated = len(prompt) - max_len
    return f"{prompt[:max_len]}\n...[truncated {truncated} chars]"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_timestamp() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _otel_endpoint_available(host: str) -> bool:
    url = f"{host.rstrip('/')}/api/public/otel/v1/traces"
    req = Request(url, method="OPTIONS")
    try:
        with urlopen(req, timeout=2) as resp:
            return resp.status != 404
    except HTTPError as exc:
        return exc.code != 404
    except URLError:
        return False


class _LegacySpan:
    def __init__(
        self,
        client: "_LegacyLangfuseClient",
        trace_id: str,
        span_id: str,
        span_type: str = "span",
    ) -> None:
        self._client = client
        self._trace_id = trace_id
        self._span_id = span_id
        self._span_type = span_type

    def start_span(self, *, name: str, input: Any = None, metadata: Any = None) -> "_LegacySpan":
        return self._client._create_span(
            trace_id=self._trace_id,
            parent_id=self._span_id,
            name=name,
            input=input,
            metadata=metadata,
        )

    def start_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any = None,
        metadata: Any = None,
        model: str | None = None,
    ) -> "_LegacySpan":
        if as_type == "generation":
            return self._client._create_generation(
                trace_id=self._trace_id,
                parent_id=self._span_id,
                name=name,
                input=input,
                metadata=metadata,
                model=model,
            )
        return self.start_span(name=name, input=input, metadata=metadata)

    def update(self, **kwargs: Any) -> None:
        self._client._update_observation(self._span_id, self._trace_id, self._span_type, **kwargs)

    def update_trace(self, **kwargs: Any) -> None:
        self._client._update_trace(self._trace_id, **kwargs)

    def end(self) -> None:
        self._client._end_observation(self._span_id, self._trace_id, self._span_type)


class _LegacyLangfuseClient:
    def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
        if FernLangfuse is None:
            raise RuntimeError("Langfuse ingestion client not available")
        self._api = FernLangfuse(
            base_url=host,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version="legacy",
            x_langfuse_public_key=public_key,
        )
        self._lock = threading.Lock()

    def create_trace_id(self) -> str:
        return str(uuid4())

    def start_span(
        self,
        *,
        trace_context: Optional[Dict[str, Any]] = None,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Any] = None,
        **_: Any,
    ) -> _LegacySpan:
        trace_id = trace_context.get("trace_id") if trace_context else self.create_trace_id()
        span_id = str(uuid4())
        trace_body = TraceBody(
            id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            timestamp=_now_utc(),
        )
        span_body = CreateSpanBody(
            id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=_now_utc(),
            input=input,
            output=output,
            metadata=metadata,
        )
        self._send_events(
            [
                IngestionEvent_TraceCreate(
                    id=str(uuid4()),
                    timestamp=_event_timestamp(),
                    body=trace_body,
                ),
                IngestionEvent_SpanCreate(
                    id=str(uuid4()),
                    timestamp=_event_timestamp(),
                    body=span_body,
                ),
            ]
        )
        return _LegacySpan(self, trace_id, span_id, "span")

    def _create_span(
        self,
        *,
        trace_id: str,
        parent_id: Optional[str],
        name: str,
        input: Any = None,
        metadata: Any = None,
    ) -> _LegacySpan:
        span_id = str(uuid4())
        span_body = CreateSpanBody(
            id=span_id,
            trace_id=trace_id,
            parent_observation_id=parent_id,
            name=name,
            start_time=_now_utc(),
            input=input,
            metadata=metadata,
        )
        self._send_events(
            [
                IngestionEvent_SpanCreate(
                    id=str(uuid4()),
                    timestamp=_event_timestamp(),
                    body=span_body,
                )
            ]
        )
        return _LegacySpan(self, trace_id, span_id, "span")

    def _create_generation(
        self,
        *,
        trace_id: str,
        parent_id: Optional[str],
        name: str,
        input: Any = None,
        metadata: Any = None,
        model: Optional[str] = None,
    ) -> _LegacySpan:
        gen_id = str(uuid4())
        gen_body = CreateGenerationBody(
            id=gen_id,
            trace_id=trace_id,
            parent_observation_id=parent_id,
            name=name,
            start_time=_now_utc(),
            input=input,
            metadata=metadata,
            model=model,
        )
        self._send_events(
            [
                IngestionEvent_GenerationCreate(
                    id=str(uuid4()),
                    timestamp=_event_timestamp(),
                    body=gen_body,
                )
            ]
        )
        return _LegacySpan(self, trace_id, gen_id, "generation")

    def _update_trace(self, trace_id: str, **kwargs: Any) -> None:
        trace_body = TraceBody(id=trace_id, timestamp=_now_utc(), **kwargs)
        self._send_events(
            [
                IngestionEvent_TraceCreate(
                    id=str(uuid4()),
                    timestamp=_event_timestamp(),
                    body=trace_body,
                )
            ]
        )

    def _update_observation(self, span_id: str, trace_id: str, span_type: str, **kwargs: Any) -> None:
        if span_type == "generation":
            body = UpdateGenerationBody(id=span_id, trace_id=trace_id, **kwargs)
            event = IngestionEvent_GenerationUpdate(
                id=str(uuid4()),
                timestamp=_event_timestamp(),
                body=body,
            )
        else:
            body = UpdateSpanBody(id=span_id, trace_id=trace_id, **kwargs)
            event = IngestionEvent_SpanUpdate(
                id=str(uuid4()),
                timestamp=_event_timestamp(),
                body=body,
            )
        self._send_events([event])

    def _end_observation(self, span_id: str, trace_id: str, span_type: str) -> None:
        if span_type == "generation":
            body = UpdateGenerationBody(id=span_id, trace_id=trace_id, end_time=_now_utc())
            event = IngestionEvent_GenerationUpdate(
                id=str(uuid4()),
                timestamp=_event_timestamp(),
                body=body,
            )
        else:
            body = UpdateSpanBody(id=span_id, trace_id=trace_id, end_time=_now_utc())
            event = IngestionEvent_SpanUpdate(
                id=str(uuid4()),
                timestamp=_event_timestamp(),
                body=body,
            )
        self._send_events([event])

    def _send_events(self, events: List[Any]) -> None:
        if not events:
            return
        try:
            with self._lock:
                self._api.ingestion.batch(batch=events)
        except Exception:
            # Keep chat flow alive even if ingestion fails.
            return


def _build_prompt_source(history: List[str], user_text: str) -> str:
    if not history:
        return user_text
    return "\n".join(history + [f"User: {user_text}"])


def _build_retrieval_source(history: List[str], user_text: str, max_turns: int = 1) -> str:
    if not history:
        return user_text
    if max_turns <= 0:
        return user_text
    take = max_turns * 2
    context = history[-take:]
    return "\n".join(context + [f"User: {user_text}"])


def _build_llm(config: ModelConfig) -> LLM:
    if config.provider == "ollama":
        return OllamaLLM(config)
    if config.provider == "deepseek":
        return DeepSeekLLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _build_embedder(config: EmbeddingConfig) -> EmbeddingClient:
    if config.provider == "ollama":
        return OllamaEmbeddings(config)
    if config.provider == "deepseek":
        return DeepSeekEmbeddings(config)
    if config.provider == "hf":
        return HFLocalEmbeddings(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def _build_extractor(config: PreferenceConfig, prefer_model: bool) -> PreferenceExtractor:
    return build_preference_extractor(config, prefer_model=prefer_model)


def _get_completion_tokens(llm: LLM) -> int | None:
    usage = getattr(llm, "last_usage", None)
    if usage is None:
        return None
    return usage.completion_tokens


@dataclass
class ChatSession:
    tree: HierarchicalMemoryTree
    llm: LLM
    extractor: PreferenceExtractor
    pref_config: PreferenceConfig
    extractor_state: ExtractionState = field(default_factory=ExtractionState)
    history_lines: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_debug: Dict[str, Any] | None = field(default=None, init=False)
    session_id: str = field(default_factory=lambda: uuid4().hex)

    def _is_short_query(self, user_text: str) -> bool:
        threshold = self.tree.retrieval_config.short_query_max_chars
        return len(user_text.strip()) <= threshold

    def _should_use_context_retrieval(self, user_text: str, trace: Dict[str, Any] | None) -> bool:
        if not self.history_lines:
            return False
        if not self._is_short_query(user_text):
            return False
        if not trace:
            return False
        return trace.get("strategy") == "fallback"

    def respond_with_progress(
        self,
        user_text: str,
        progress_cb: callable,
        use_tree: bool = True,
        use_context: bool = True,
    ) -> str:
        global _langfuse_client

        # Create langfuse trace early to capture the full process
        trace = None
        trace_turn = None
        if _langfuse_client is not None:
            trace_turn = len(self.messages) // 2 + 1
            # Initial trace input (will be updated later with full prompt)
            trace_input = {"user_text": user_text, "use_tree": use_tree, "use_context": use_context}
            trace = _langfuse_client.start_span(
                name="chat_response",
                trace_context={"trace_id": _langfuse_client.create_trace_id()},
                input=trace_input,
                metadata={"turn": trace_turn},
            )
            trace.update_trace(
                name="chat_response",
                session_id=self.session_id,
                input=trace_input,
                metadata={"turn": trace_turn},
            )

        mode = "memory_tree" if use_tree else "baseline"
        progress: Dict[str, Any] = {
            "stage": "start",
            "timing_ms": {},
            "mode": mode,
            "use_context": use_context,
        }
        progress_cb(dict(progress))
        total_start = perf_counter()
        prompt_source = _build_prompt_source(self.history_lines, user_text) if use_context else user_text
        pref_ms = None
        retrieval_trace = None
        fusion = None
        memory_info = None
        if use_tree:
            pref_start = perf_counter()

            # Langfuse span for preference retrieval
            retrieval_span = None
            if trace is not None:
                retrieval_span = trace.start_span(
                    name="preference_retrieval",
                    input={"query": user_text},
                )

            fusion, retrieval_trace = self.tree.get_preference_trace(user_text)
            pref_ms = (perf_counter() - pref_start) * 1000
            if self._should_use_context_retrieval(user_text, retrieval_trace):
                retrieval_text = _build_retrieval_source(self.history_lines, user_text, max_turns=1)
                context_start = perf_counter()
                fusion, retrieval_trace = self.tree.get_preference_trace(retrieval_text)
                pref_ms += (perf_counter() - context_start) * 1000
                if retrieval_trace is not None:
                    retrieval_trace["context_used"] = True

            if retrieval_span is not None:
                retrieval_span.update(
                    output={"strategy": retrieval_trace.get("strategy") if retrieval_trace else None},
                    metadata={"duration_ms": pref_ms},
                )
                retrieval_span.end()

            progress["stage"] = "preference_retrieval"
            progress["timing_ms"]["preference_retrieval"] = pref_ms
            progress["retrieval"] = retrieval_trace
            memory_info = {
                "fusion": self.tree._fusion_payload(fusion),
                "path": retrieval_trace.get("selected_path") if retrieval_trace else None,
                "strategy": retrieval_trace.get("strategy") if retrieval_trace else None,
            }
            progress["memory"] = memory_info
            progress_cb(dict(progress))

        prompt_start = perf_counter()
        if fusion is None:
            prompt = prompt_source
        else:
            prompt = build_prompt(prompt_source, fusion, self.pref_config)
        prompt_ms = (perf_counter() - prompt_start) * 1000

        # Update trace input with the full prompt
        if trace is not None:
            trace_input = {
                "user_text": user_text,
                "use_tree": use_tree,
                "use_context": use_context,
                "prompt": prompt,
                "prompt_source": prompt_source,
                "has_fusion": fusion is not None
            }
            trace.update(input=trace_input)
            trace.update_trace(input=trace_input)
        if _env_flag("PAMT_PROMPT_LOG"):
            try:
                max_len = int(os.environ.get("PAMT_PROMPT_LOG_MAX", "2000") or 2000)
            except ValueError:
                max_len = 2000
            prompt_mode = "memory" if fusion is not None else "raw"
            logger.info("LLM prompt (%s):\n%s", prompt_mode, _prompt_preview(prompt, max_len=max_len))
        progress["stage"] = "prompt_build"
        progress["timing_ms"]["prompt_build"] = prompt_ms
        progress_cb(dict(progress))

        progress["stage"] = "llm_generate"
        progress_cb(dict(progress))

        # Langfuse generation for LLM call
        generation = None
        if trace is not None:
            generation = trace.start_observation(
                name="llm_generate",
                as_type="generation",
                model=getattr(self.llm, "config", None) and self.llm.config.model_name or "unknown",
                input=prompt,
            )

        llm_start = perf_counter()
        response = self.llm.generate(prompt)
        token_count = _get_completion_tokens(self.llm)
        llm_ms = (perf_counter() - llm_start) * 1000

        # End langfuse generation
        if generation is not None:
            usage = getattr(self.llm, "last_usage", None)
            usage_details = None
            if usage is not None:
                usage_details = {
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                }
            generation.update(
                output=response,
                usage_details=usage_details,
                metadata={"duration_ms": llm_ms},
            )
            generation.end()

        progress["timing_ms"]["llm_generate"] = llm_ms
        progress_cb(dict(progress))

        self.history_lines.append(f"User: {user_text}")
        self.history_lines.append(f"Assistant: {response}")
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": response})

        extract_ms = None
        route_ms = None
        update_ms = None
        update_info = None
        if use_tree:
            # Langfuse span for extraction
            extract_span = None
            if trace is not None:
                extract_span = trace.start_span(
                    name="preference_extraction",
                    input={"user": user_text, "response": response[:200]},
                )

            extract_start = perf_counter()
            pref = self.extractor.extract(
                user_text,
                response,
                self.extractor_state,
                token_count,
            )
            content_text = self.tree._combine_text(user_text, response)
            extract_ms = (perf_counter() - extract_start) * 1000

            if extract_span is not None:
                extract_span.update(
                    output={"preference": pref.__dict__ if hasattr(pref, "__dict__") else str(pref)},
                    metadata={"duration_ms": extract_ms},
                )
                extract_span.end()

            progress["stage"] = "extractor"
            progress["timing_ms"]["extractor"] = extract_ms
            progress_cb(dict(progress))

            # Langfuse span for routing
            route_span = None
            if trace is not None:
                route_span = trace.start_span(
                    name="route_query",
                    input={"user": user_text},
                )

            route_start = perf_counter()
            path = self.tree.route_response(user_text, response)
            route_ms = (perf_counter() - route_start) * 1000

            if route_span is not None:
                route_span.update(
                    output={"path": path},
                    metadata={"duration_ms": route_ms},
                )
                route_span.end()

            progress["stage"] = "route_query"
            progress["timing_ms"]["route_query"] = route_ms
            progress_cb(dict(progress))

            # Langfuse span for update
            update_span = None
            if trace is not None:
                update_span = trace.start_span(
                    name="memory_update",
                    input={"path": path},
                )

            update_start = perf_counter()
            update_info = self.tree.update_preference_trace(path, pref, content_text)
            update_ms = (perf_counter() - update_start) * 1000

            if update_span is not None:
                update_span.update(
                    output=update_info,
                    metadata={"duration_ms": update_ms},
                )
                update_span.end()

            progress["stage"] = "update"
            progress["timing_ms"]["update"] = update_ms
            progress["update"] = update_info
            progress_cb(dict(progress))

        total_ms = (perf_counter() - total_start) * 1000
        self.last_debug = {
            "timing_ms": {
                "total": total_ms,
                "preference_retrieval": pref_ms,
                "prompt_build": prompt_ms,
                "llm_generate": llm_ms,
                "extractor": extract_ms,
                "route_query": route_ms,
                "update": update_ms,
            },
            "retrieval": retrieval_trace,
            "update": update_info,
            "mode": mode,
            "use_context": use_context,
            "memory": memory_info,
        }

        # End langfuse trace
        if trace is not None:
            trace.update(
                output={"response": response},
                metadata={"total_ms": total_ms, "mode": mode},
            )
            trace.update_trace(
                output={"response": response},
                metadata={"total_ms": total_ms, "mode": mode},
            )
            trace.end()

        progress["stage"] = "done"
        progress["timing_ms"]["total"] = total_ms
        progress_cb(dict(progress))
        return response


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "PAMTChat/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/app.css":
            self._serve_static("app.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/api/state":
            payload = self._build_state()
            self._send_json(payload)
            return
        if parsed.path == "/api/job":
            qs = parse_qs(parsed.query)
            job_id = qs.get("job_id", [""])[0]
            if not job_id:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing job_id")
                return
            with self.server.jobs_lock:
                job = self.server.jobs.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                return
            self._send_json(job)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        message = str(data.get("message", "")).strip()
        if not message:
            self.send_error(HTTPStatus.BAD_REQUEST, "Empty message")
            return
        use_tree = data.get("use_tree", True)
        if isinstance(use_tree, str):
            use_tree = use_tree.strip().lower() not in {"0", "false", "no", "off"}
        use_tree = bool(use_tree)
        use_context = data.get("use_context", True)
        if isinstance(use_context, str):
            use_context = use_context.strip().lower() not in {"0", "false", "no", "off"}
        use_context = bool(use_context)
        job_id = uuid4().hex
        with self.server.jobs_lock:
            self.server.jobs[job_id] = {
                "status": "running",
                "progress": {"stage": "queued", "timing_ms": {}},
            }
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, message, use_tree, use_context),
            daemon=True,
        )
        thread.start()
        self._send_json({"job_id": job_id})

    def _run_job(self, job_id: str, message: str, use_tree: bool, use_context: bool) -> None:
        def progress_cb(update: Dict[str, Any]) -> None:
            with self.server.jobs_lock:
                job = self.server.jobs.get(job_id)
                if not job:
                    return
                job["progress"] = update

        try:
            with self.server.state_lock:
                response = self.server.session.respond_with_progress(
                    message,
                    progress_cb,
                    use_tree=use_tree,
                    use_context=use_context,
                )
                payload = self._build_state(response)
            with self.server.jobs_lock:
                job = self.server.jobs.get(job_id)
                if not job:
                    return
                job.update(payload)
                job["status"] = "done"
        except Exception as exc:
            with self.server.jobs_lock:
                job = self.server.jobs.get(job_id)
                if not job:
                    return
                job["status"] = "error"
                job["error"] = str(exc)

    def _build_state(self, response: str | None = None) -> Dict[str, Any]:
        tree_snapshot = self.server.session.tree.snapshot()
        payload = {
            "messages": self.server.session.messages,
            "tree": tree_snapshot,
            "debug": self.server.session.last_debug,
        }
        if response is not None:
            payload["response"] = response
        return payload

    def _serve_static(self, name: str, content_type: str) -> None:
        web_dir = self.server.web_dir
        path = web_dir / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ChatServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[ChatHandler], session: ChatSession, web_dir: Path):
        super().__init__(server_address, handler)
        self.session = session
        self.web_dir = web_dir
        self.state_lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAMT memory-tree visual chatbot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)

    parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "deepseek"])
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--deepseek-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--deepseek-key", default="")

    parser.add_argument("--embed-provider", default="ollama", choices=["ollama", "deepseek", "hf"])
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--embed-ollama-url", default="http://localhost:11434/api/embeddings")

    parser.add_argument("--strict", type=float, default=0.75)
    parser.add_argument("--loose", type=float, default=0.6)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument(
        "--use-model-extractor",
        dest="use_model_extractor",
        action="store_true",
        help="Prefer model-backed extractor (default).",
    )
    parser.add_argument(
        "--heuristic-extractor",
        dest="use_model_extractor",
        action="store_false",
        help="Force heuristic extractor.",
    )
    parser.set_defaults(use_model_extractor=True)

    parser.add_argument("--tone-model-id", default=None)
    parser.add_argument("--emotion-model-id", default=None)
    parser.add_argument("--formality-model-id", default=None)
    parser.add_argument("--density-model-id", default=None)
    parser.add_argument("--spacy-model", default=None)
    parser.add_argument("--tone-max-length", type=int, default=None)
    parser.add_argument("--formality-max-length", type=int, default=None)
    parser.add_argument("--opennre-max-pairs", type=int, default=None)
    parser.add_argument("--opennre-max-entities", type=int, default=None)
    parser.add_argument("--opennre-max-text-tokens", type=int, default=None)

    # Langfuse arguments
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Enable Langfuse tracing (requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY env vars).",
    )
    parser.add_argument("--langfuse-host", default="http://127.0.0.1:3000", help="Langfuse host URL.")
    return parser.parse_args()


def main() -> None:
    global _langfuse_client

    setup_logging()
    logger = logging.getLogger(__name__)
    args = _parse_args()
    web_dir = Path(__file__).resolve().parent / "web"

    # Initialize Langfuse if enabled
    if args.langfuse:
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse requested but not installed. Run: pip install langfuse")
        else:
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
            if not public_key or not secret_key:
                logger.warning(
                    "Langfuse enabled but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
                    "Tracing will be disabled."
                )
            else:
                if _otel_endpoint_available(args.langfuse_host):
                    _langfuse_client = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=args.langfuse_host,
                    )
                    logger.info("Langfuse OTEL tracing enabled (host=%s)", args.langfuse_host)
                else:
                    _langfuse_client = _LegacyLangfuseClient(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=args.langfuse_host,
                    )
                    logger.warning(
                        "Langfuse OTEL endpoint missing at %s; using legacy ingestion.",
                        args.langfuse_host,
                    )

    model_config = ModelConfig(
        provider=args.llm_provider,
        model_name=args.llm_model,
        ollama_url=args.ollama_url,
        api_base_url=args.deepseek_base,
        api_key=args.deepseek_key,
    )
    embed_config = EmbeddingConfig(
        provider=args.embed_provider,
        model_name=args.embed_model,
        ollama_url=args.embed_ollama_url,
        api_base_url=args.deepseek_base,
        api_key=args.deepseek_key,
    )
    retrieval_config = RetrievalConfig(
        similarity_strict=args.strict,
        similarity_loose=args.loose,
        max_candidates=args.max_candidates,
    )

    pref_config = PreferenceConfig()
    model_cfg = pref_config.preference_models
    if args.tone_model_id:
        model_cfg.tone_model_id = args.tone_model_id
    if args.emotion_model_id:
        model_cfg.emotion_model_id = args.emotion_model_id
    if args.formality_model_id:
        model_cfg.formality_model_id = args.formality_model_id
    if args.density_model_id:
        model_cfg.density_model_id = args.density_model_id
    if args.spacy_model:
        model_cfg.spacy_model = args.spacy_model
    if args.tone_max_length is not None:
        model_cfg.tone_max_length = args.tone_max_length
    if args.formality_max_length is not None:
        model_cfg.formality_max_length = args.formality_max_length
    if args.opennre_max_pairs is not None:
        model_cfg.opennre_max_pairs = args.opennre_max_pairs
    if args.opennre_max_entities is not None:
        model_cfg.opennre_max_entities = args.opennre_max_entities
    if args.opennre_max_text_tokens is not None:
        model_cfg.opennre_max_text_tokens = args.opennre_max_text_tokens
    update_config = UpdateConfig()

    llm = _build_llm(model_config)
    embedder = _build_embedder(embed_config)
    extractor = _build_extractor(pref_config, args.use_model_extractor)
    logger.info(
        "visual_chatbot: llm_provider=%s llm_model=%s embed_provider=%s",
        model_config.provider,
        model_config.model_name,
        embed_config.provider,
    )

    tree = HierarchicalMemoryTree(
        update_config=update_config,
        retrieval_config=retrieval_config,
        embedder=embedder,
        llm=llm,
        label_prompt_path="prompts/category_leaf.txt",
        leaf_prompt_path="prompts/leaf_only.txt",
    )

    session = ChatSession(tree=tree, llm=llm, extractor=extractor, pref_config=pref_config)

    server = ChatServer((args.host, args.port), ChatHandler, session, web_dir)
    print(f"Visual chatbot running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
