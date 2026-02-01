from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

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
            fusion, retrieval_trace = self.tree.get_preference_trace(user_text)
            pref_ms = (perf_counter() - pref_start) * 1000
            if self._should_use_context_retrieval(user_text, retrieval_trace):
                retrieval_text = _build_retrieval_source(self.history_lines, user_text, max_turns=1)
                context_start = perf_counter()
                fusion, retrieval_trace = self.tree.get_preference_trace(retrieval_text)
                pref_ms += (perf_counter() - context_start) * 1000
                if retrieval_trace is not None:
                    retrieval_trace["context_used"] = True
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
        progress["stage"] = "prompt_build"
        progress["timing_ms"]["prompt_build"] = prompt_ms
        progress_cb(dict(progress))

        progress["stage"] = "llm_generate"
        progress_cb(dict(progress))
        llm_start = perf_counter()
        response = self.llm.generate(prompt)
        token_count = _get_completion_tokens(self.llm)
        llm_ms = (perf_counter() - llm_start) * 1000
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
            extract_start = perf_counter()
            pref = self.extractor.extract(
                user_text,
                response,
                self.extractor_state,
                token_count,
            )
            content_text = self.tree._combine_text(user_text, response)
            extract_ms = (perf_counter() - extract_start) * 1000
            progress["stage"] = "extractor"
            progress["timing_ms"]["extractor"] = extract_ms
            progress_cb(dict(progress))

            route_start = perf_counter()
            path = self.tree.route_response(user_text, response)
            route_ms = (perf_counter() - route_start) * 1000
            progress["stage"] = "route_query"
            progress["timing_ms"]["route_query"] = route_ms
            progress_cb(dict(progress))

            update_start = perf_counter()
            update_info = self.tree.update_preference_trace(path, pref, content_text)
            update_ms = (perf_counter() - update_start) * 1000
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
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = _parse_args()
    web_dir = Path(__file__).resolve().parent / "web"

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



