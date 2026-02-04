from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from pamt.config import EmbeddingConfig, ModelConfig, PreferenceConfig, UpdateConfig
from pamt.core.memory_tree import RetrievalConfig
from pamt.llms.models import DeepSeekLLM, LLM
from pamt.memory_plugin import MemoryPromptPlugin, create_memory_plugin
from pamt.logging_utils import setup_logging


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
    parser = argparse.ArgumentParser(description="PAMT memory-tree chatbot example (DeepSeek + HF embeddings).")
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--deepseek-key", default="")

    parser.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--memory-file", default="data/pamt_memory.json")

    parser.add_argument("--strict", type=float, default=0.75)
    parser.add_argument("--loose", type=float, default=0.6)
    parser.add_argument("--max-candidates", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = _parse_args()

    model_config = ModelConfig(
        model_name=args.llm_model,
        api_base_url=args.deepseek_base,
        api_key=args.deepseek_key,
    )
    embed_config = EmbeddingConfig(
        model_name=args.embed_model,
        hf_device=args.embed_device,
    )
    retrieval_config = RetrievalConfig(
        similarity_strict=args.strict,
        similarity_loose=args.loose,
        max_candidates=args.max_candidates,
    )

    pref_config = PreferenceConfig()
    update_config = UpdateConfig()

    llm = DeepSeekLLM(model_config)
    logger.info(
        "chatbot: llm_model=%s embed_model=%s memory_file=%s",
        model_config.model_name,
        embed_config.model_name,
        args.memory_file,
    )
    plugin = create_memory_plugin(
        model_config=model_config,
        embedding_config=embed_config,
        preference_config=pref_config,
        update_config=update_config,
        retrieval_config=retrieval_config,
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
