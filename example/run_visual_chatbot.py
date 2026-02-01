from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "config" / "api_config.json"
    if candidate.is_file():
        return candidate
    return repo_root / "config" / "api_config.example.json"


def _extract_config_path(argv: list[str]) -> tuple[Path, list[str]]:
    if not argv:
        return _default_config_path(), []
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 >= len(argv):
            print("Missing value for --config")
            raise SystemExit(1)
        path = Path(argv[idx + 1])
        remaining = argv[:idx] + argv[idx + 2 :]
        return path, remaining
    for arg in argv:
        if arg.startswith("--config="):
            _, value = arg.split("=", 1)
            remaining = [item for item in argv if item != arg]
            return Path(value), remaining
    return _default_config_path(), argv


def _load_config(path: Path) -> dict:
    if not path.is_file():
        print(f"Config not found: {path}")
        raise SystemExit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config: {path} ({exc})")
        raise SystemExit(1)


def _default_embed_model(provider: str) -> str:
    if provider == "deepseek":
        return "deepseek-embedding"
    if provider == "hf":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return "nomic-embed-text"


def _build_args_from_config(config: dict) -> list[str]:
    provider = str(config.get("provider", "ollama"))
    llm_model = str(config.get("model_name", "qwen2.5:3b"))
    base_url = str(config.get("api_base_url", "https://api.deepseek.com/v1"))
    api_key = str(config.get("api_key", ""))
    ollama_url = str(config.get("ollama_url", "http://localhost:11434/api/generate"))

    embed_provider = str(config.get("embed_provider", provider))
    embed_model = str(config.get("embed_model_name", config.get("embed_model", ""))).strip()
    if not embed_model:
        embed_model = _default_embed_model(embed_provider)
    embed_ollama_url = str(
        config.get("embed_ollama_url", "http://localhost:11434/api/embeddings")
    )

    return [
        "visual_chatbot.py",
        "--llm-provider",
        provider,
        "--embed-provider",
        embed_provider,
        "--llm-model",
        llm_model,
        "--embed-model",
        embed_model,
        "--deepseek-base",
        base_url,
        "--deepseek-key",
        api_key,
        "--ollama-url",
        ollama_url,
        "--embed-ollama-url",
        embed_ollama_url,
    ]


def main() -> None:
    script_path = Path(__file__).resolve().parent / "visual_chatbot.py"
    config_path, remaining = _extract_config_path(sys.argv[1:])
    if remaining:
        sys.argv = ["visual_chatbot.py", *remaining]
    else:
        config = _load_config(config_path)
        sys.argv = _build_args_from_config(config)
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
