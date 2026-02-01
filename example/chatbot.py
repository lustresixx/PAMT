from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from pamt.config import EmbeddingConfig, ModelConfig, PreferenceConfig, UpdateConfig
from pamt.core.memory_tree import RetrievalConfig
from pamt.llms.models import DeepSeekLLM, LLM, OllamaLLM
from pamt.memory_plugin import MemoryPromptPlugin, create_memory_plugin
from pamt.logging_utils import setup_logging


def _build_llm(config: ModelConfig) -> LLM:
    if config.provider == "ollama":
        return OllamaLLM(config)
    if config.provider == "deepseek":
        return DeepSeekLLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _get_completion_tokens(llm: LLM) -> int | None:
    usage = getattr(llm, "last_usage", None)
    if usage is None:
        return None
    return usage.completion_tokens


@dataclass
class ChatSession:
    llm: LLM
    plugin: MemoryPromptPlugin

    def respond(self, user_text: str) -> str:
        augmentation = self.plugin.augment(user_text)
        prompt = augmentation.prompt
        response = self.llm.generate(prompt)
        token_count = _get_completion_tokens(self.llm)
        self.plugin.update(user_text, response, token_count)
        return response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAMT memory-tree chatbot example.")
    parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "deepseek"])
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--deepseek-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--deepseek-key", default="")

    parser.add_argument("--embed-provider", default="ollama", choices=["ollama", "deepseek", "hf"])
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--embed-ollama-url", default="http://localhost:11434/api/embeddings")
    parser.add_argument("--memory-file", default="data/pamt_memory.json")

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
    logger.info(
        "chatbot: llm_provider=%s llm_model=%s embed_provider=%s memory_file=%s",
        model_config.provider,
        model_config.model_name,
        embed_config.provider,
        args.memory_file,
    )
    plugin = create_memory_plugin(
        model_config=model_config,
        embedding_config=embed_config,
        preference_config=pref_config,
        update_config=update_config,
        retrieval_config=retrieval_config,
        prefer_model_extractor=args.use_model_extractor,
        storage_path=args.memory_file,
    )
    session = ChatSession(llm=llm, plugin=plugin)

    print("PAMT chatbot ready. Type 'exit' to quit.")
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        response = session.respond(user_text)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()



