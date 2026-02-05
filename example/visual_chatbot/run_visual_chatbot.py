from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
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
        return json.loads(path.read_text(encoding="utf-8-sig"))
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

    args = [
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
    pref_models = config.get("preference_models", {}) or {}
    pref_map = [
        ("--tone-model-id", pref_models.get("tone_model_id")),
        ("--emotion-model-id", pref_models.get("emotion_model_id")),
        ("--formality-model-id", pref_models.get("formality_model_id")),
        ("--density-model-id", pref_models.get("density_model_id")),
        ("--spacy-model", pref_models.get("spacy_model")),
        ("--tone-max-length", pref_models.get("tone_max_length")),
        ("--formality-max-length", pref_models.get("formality_max_length")),
        ("--opennre-max-pairs", pref_models.get("opennre_max_pairs")),
        ("--opennre-max-entities", pref_models.get("opennre_max_entities")),
        ("--opennre-max-text-tokens", pref_models.get("opennre_max_text_tokens")),
    ]
    for flag, value in pref_map:
        if value is None or value == "":
            continue
        args.extend([flag, str(value)])

    # Add langfuse arguments if enabled
    if config.get("langfuse_enabled", False):
        args.append("--langfuse")
        langfuse_host = config.get("langfuse_host", "http://127.0.0.1:3000")
        args.extend(["--langfuse-host", langfuse_host])
    return args


def main() -> None:
    script_path = Path(__file__).resolve().parent / "visual_chatbot.py"

    config_path, remaining = _extract_config_path(sys.argv[1:])
    config = _load_config(config_path)

    # Check for --langfuse flag in remaining args or config
    langfuse_enabled = config.get("langfuse_enabled", False)

    # Note: --start-langfuse is no longer supported here.
    # Use `python start_langfuse.py` to start Langfuse Docker services separately.
    if "--start-langfuse" in remaining:
        print("[Warning] --start-langfuse is deprecated. Use `python start_langfuse.py` instead.")
        remaining = [arg for arg in remaining if arg != "--start-langfuse"]

    # Handle langfuse environment setup
    if langfuse_enabled or "--langfuse" in remaining:
        # Set environment variables for langfuse
        langfuse_public_key = config.get("langfuse_public_key", "")
        langfuse_secret_key = config.get("langfuse_secret_key", "")

        if langfuse_public_key and langfuse_public_key != "YOUR_LANGFUSE_PUBLIC_KEY":
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
        if langfuse_secret_key and langfuse_secret_key != "YOUR_LANGFUSE_SECRET_KEY":
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret_key


    if remaining:
        sys.argv = ["visual_chatbot.py", *remaining]
    else:
        sys.argv = _build_args_from_config(config)

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
