import os

os.environ.setdefault("PAMT_REAL_TESTS", "1")
os.environ.setdefault("HF_HOME", "data/hf_cache")
if not os.environ.get("PAMT_HF_TOKEN"):
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    os.environ.pop("HF_TOKEN", None)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import unittest

from pamt.config import EmbeddingConfig, ModelConfig, PreferenceConfig, UpdateConfig
from pamt.core.memory_tree import HierarchicalMemoryTree, RetrievalConfig
from pamt.core.update import PAMTState, PAMTUpdater
from pamt.extractors.preference_extractor import ExtractionState, ModelPreferenceExtractor
from pamt.llms.models import DeepSeekLLM
from pamt.embeddings.models import HFLocalEmbeddings


def _device_hint():
    try:
        import torch
    except Exception:
        return None, "cpu"
    if torch.cuda.is_available():
        return 0, "cuda"
    return None, "cpu"


def _print_distribution(name: str, text: str, labels: list[str], probs: list[float]) -> None:
    pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]
    summary = ", ".join(f"{label}={score:.3f}" for label, score in pairs)
    print(f"[{name}] {summary} | text={text!r}")


def _print_triples(name: str, triples: list[tuple[str, str, str]]) -> None:
    preview = triples[:5]
    print(f"[{name}] triples={preview} (count={len(triples)})")


class RealModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        device_id, device_name = _device_hint()
        print(f"[device] preference_models={device_name}")

        cls.config = PreferenceConfig()
        cls.extractor = ModelPreferenceExtractor.from_preferred_models(
            cls.config,
            device=device_id,
            hf_cache_dir=os.environ.get("PAMT_HF_CACHE_DIR"),
        )
        cls.updater = PAMTUpdater(UpdateConfig())

    def test_preference_extractor_bilingual(self) -> None:
        cases = [
            (
                "The manual is confusing and I feel disappointed.",
                "We will improve the guide for the Apple device in San Francisco and provide clearer steps by 2025.",
                24,
            ),
            (
                "请尽快更新说明书，我对目前的内容很失望。",
                "很抱歉给你带来不便，我们会提供更清晰的指引。",
                18,
            ),
            (
                "Thanks for the quick response, everything is working now!",
                "Happy to help. Let us know if you need anything else.",
                26,
            ),
            (
                "这个功能让我很开心，体验很好，谢谢！",
                "感谢你的反馈，我们会继续优化。",
                14,
            ),
            (
                "Although the update fixed the crashes, the interface is still cluttered, so I need a clearer, step-by-step walkthrough.",
                "We appreciate the detailed feedback and will deliver a redesigned guide with annotated screenshots and a clearer navigation flow.",
                42,
            ),
            (
                "虽然这次更新减少了崩溃，但界面仍然复杂，我希望能看到更清晰、分步骤的教程来完成配置。",
                "感谢你的耐心反馈，我们会补充图文并茂的操作说明，并在下个版本中优化界面层级。",
                36,
            ),
        ]

        for user_text, assistant_text, token_count in cases:
            state = ExtractionState()
            pref = self.extractor.extract(user_text, assistant_text, state, assistant_token_count=token_count)

            tone_idx, tone_probs = pref.tone
            emo_idx, emo_probs = pref.emotion
            print(f"[input] user={user_text!r}")
            print(f"[input] assistant={assistant_text!r}")
            print(
                f"[pref] tone={self.config.tone_labels[tone_idx]} "
                f"emotion={self.config.emotion_labels[emo_idx]} "
                f"len={pref.length:.3f} den={pref.density:.3f} form={pref.formality:.3f}"
            )
            _print_distribution("tone", user_text, self.config.tone_labels, tone_probs)
            _print_distribution("emotion", user_text, self.config.emotion_labels, emo_probs)

            self.assertEqual(len(tone_probs), len(self.config.tone_labels))
            self.assertEqual(len(emo_probs), len(self.config.emotion_labels))
            self.assertTrue(0 <= tone_idx < len(self.config.tone_labels))
            self.assertTrue(0 <= emo_idx < len(self.config.emotion_labels))
            self.assertAlmostEqual(pref.length, token_count / 300.0, places=4)
            self.assertGreaterEqual(pref.density, 0.0)
            self.assertLessEqual(pref.density, 1.0)
            self.assertGreaterEqual(pref.formality, 0.0)
            self.assertLessEqual(pref.formality, 1.0)

    def test_model_component_outputs(self) -> None:
        samples = [
            "Apple released the iPhone in San Francisco on 2025-01-10.",
            "Barack Obama was born in Hawaii and served as President.",
            "Please provide a concise summary of the quarterly report.",
        ]

        for text in samples:
            tone_idx, tone_probs = self.extractor.tone_model(text)
            emo_idx, emo_probs = self.extractor.emotion_model(text)
            formality = self.extractor.formality_model(text)
            triples = self.extractor.openie_model(text)
            density = min(len(triples) / max(len(text.split()), 1), 1.0)

            print(f"[component] input={text!r}")
            print(f"[component] tone={self.config.tone_labels[tone_idx]} probs={tone_probs}")
            print(f"[component] emotion={self.config.emotion_labels[emo_idx]} probs={emo_probs}")
            print(f"[component] formality={formality:.3f}")
            _print_triples("component", triples)
            print(f"[component] density={density:.3f}")

            self.assertEqual(len(tone_probs), len(self.config.tone_labels))
            self.assertEqual(len(emo_probs), len(self.config.emotion_labels))
            self.assertTrue(0 <= tone_idx < len(self.config.tone_labels))
            self.assertTrue(0 <= emo_idx < len(self.config.emotion_labels))
            self.assertGreaterEqual(formality, 0.0)
            self.assertLessEqual(formality, 1.0)
            self.assertIsInstance(triples, list)

    def test_memory_tree_real_llm_labels(self) -> None:
        # Memory tree 必须走 DeepSeek + 真实 embedding。
        api_key = os.environ.get("PAMT_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        model_config = ModelConfig(api_key=api_key)
        if not model_config.api_key:
            raise RuntimeError("DeepSeek api_key is required for memory tree labeling.")

        _device_id, device_name = _device_hint()
        print(f"[device] embeddings={device_name}")
        embed_config = EmbeddingConfig(
            hf_device=device_name,
            hf_cache_dir=os.environ.get("PAMT_HF_CACHE_DIR"),
        )

        llm = DeepSeekLLM(model_config)
        embedder = HFLocalEmbeddings(embed_config)
        tree = HierarchicalMemoryTree(
            update_config=UpdateConfig(),
            retrieval_config=RetrievalConfig(leaf_reuse_threshold=1.1),
            embedder=embedder,
            llm=llm,
        )

        samples = [
            (
                "How do I reset my device?",
                "Press and hold the reset button for five seconds.",
            ),
            (
                "Please send the invoice for last month.",
                "Sure, I will email the invoice shortly.",
            ),
            (
                "请帮我看一下说明书有没有更新。",
                "我们会尽快更新说明书并补充示例。",
            ),
        ]

        for query, response in samples:
            path = tree.route_response(query, response)
            print(f"[memory_tree] query={query!r}")
            print(f"[memory_tree] response={response!r}")
            print(f"[memory_tree] path={path}")
            self.assertEqual(len(path), 2)
            self.assertTrue(path[0])
            self.assertTrue(path[1])

    def test_sw_ema_fusion_over_three_responses(self) -> None:
        # SW/EMA 更新过程：打印每轮融合结果。
        state = PAMTState()
        user_text = "I am unhappy with the current manual and need help."
        responses = [
            "We will rewrite the guide and add screenshots.",
            "Let's schedule a call to walk you through the setup steps.",
            "Here is a quick-start checklist for troubleshooting.",
        ]

        for idx, response in enumerate(responses, start=1):
            pref = self.extractor.extract(user_text, response, ExtractionState())
            fusion, _change = self.updater.update(state, pref)
            sw_len = state.sw_history["length"][-1]
            ema_len = state.ema_history["length"][-1]
            print(
                f"[update {idx}] response={response!r} "
                f"sw_len={sw_len:.4f} ema_len={ema_len:.4f} "
                f"fusion(len={fusion.length:.4f} den={fusion.density:.4f} form={fusion.formality:.4f})"
            )
            self.assertGreaterEqual(fusion.length, 0.0)
            self.assertLessEqual(fusion.length, 1.0)
            self.assertGreaterEqual(fusion.density, 0.0)
            self.assertLessEqual(fusion.density, 1.0)
            self.assertGreaterEqual(fusion.formality, 0.0)
            self.assertLessEqual(fusion.formality, 1.0)

        self.assertEqual(len(state.sw_history["length"]), 3)
        self.assertEqual(len(state.ema_history["length"]), 3)


if __name__ == "__main__":
    unittest.main()
