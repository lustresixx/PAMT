import importlib.util
import os
import unittest

from pamt.config import PreferenceConfig
from pamt.extractors.preference_extractor import ExtractionState, ModelPreferenceExtractor


class HFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("PAMT_RUN_HF_TESTS") != "1":
            raise unittest.SkipTest("Set PAMT_RUN_HF_TESTS=1 to run HuggingFace model tests.")
        for dep in ("transformers", "torch", "gliner"):
            if importlib.util.find_spec(dep) is None:
                raise unittest.SkipTest(f"Missing dependency: {dep}")
        cls.config = PreferenceConfig()
        cls.extractor = ModelPreferenceExtractor.from_huggingface(cls.config)

    def test_hf_models_end_to_end(self) -> None:
        state = ExtractionState()
        pref = self.extractor.extract(
            "Yeah right, totally works.",
            "Sure, here is a short answer.",
            state,
        )
        self.assertEqual(len(pref.tone[1]), len(self.config.tone_labels))
        self.assertEqual(len(pref.emotion[1]), len(self.config.emotion_labels))
        self.assertGreaterEqual(pref.formality, 0.0)
        self.assertLessEqual(pref.formality, 1.0)

    def test_tone_model_outputs(self) -> None:
        samples = [
            "Oh sure, that will *definitely* work.",
            "Thank you for your help.",
            "Please find the attached report.",
        ]
        for text in samples:
            idx, probs = self.extractor.tone_model(text)
            self._print_distribution("tone", text, self.config.tone_labels, probs, idx)
            self.assertEqual(len(probs), len(self.config.tone_labels))
            self.assertTrue(0 <= idx < len(self.config.tone_labels))

    def test_emotion_model_outputs(self) -> None:
        samples = [
            "I am thrilled about the results!",
            "This makes me really sad.",
            "I am furious about the delay.",
            "I'm a bit nervous about tomorrow.",
        ]
        for text in samples:
            idx, probs = self.extractor.emotion_model(text)
            self._print_distribution("emotion", text, self.config.emotion_labels, probs, idx)
            self.assertEqual(len(probs), len(self.config.emotion_labels))
            self.assertTrue(0 <= idx < len(self.config.emotion_labels))

    def test_formality_and_density_outputs(self) -> None:
        samples = [
            "Hey, thanks! I'll get back to you.",
            "Dear Sir or Madam, I am writing to inquire about your services.",
            "The Eiffel Tower is in Paris. It was completed in 1889.",
        ]
        for text in samples:
            formality = self.extractor.formality_model(text)
            triples = self.extractor.openie_model(text) if self.extractor.openie_model else []
            print(f"[formality] {formality:.3f} | text={text!r}")
            print(f"[density] triples={len(triples)} | sample={triples[:3]}")
            self.assertGreaterEqual(formality, 0.0)
            self.assertLessEqual(formality, 1.0)
            self.assertIsInstance(triples, list)

    @staticmethod
    def _print_distribution(
        name: str,
        text: str,
        labels: list[str],
        probs: list[float],
        idx: int,
        top_k: int = 3,
    ) -> None:
        pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:top_k]
        summary = ", ".join(f"{label}={score:.3f}" for label, score in pairs)
        chosen = labels[idx] if 0 <= idx < len(labels) else "unknown"
        print(f"[{name}] {chosen} | {summary} | text={text!r}")


if __name__ == "__main__":
    unittest.main()



