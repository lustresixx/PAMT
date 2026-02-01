import unittest

from pamt.config import PreferenceConfig
from pamt.extractors.preference_extractor import ExtractionState, ModelPreferenceExtractor


class ModelPreferenceExtractorTests(unittest.TestCase):
    def test_model_extractor_with_dummy_models(self) -> None:
        config = PreferenceConfig()

        def tone_model(_text: str):
            return 1, [0.0, 1.0, 0.0, 0.0]

        def emotion_model(_text: str):
            return 0, [1.0, 0.0, 0.0, 0.0, 0.0]

        def formality_model(_text: str) -> float:
            return 0.75

        def openie_model(_text: str):
            return [("alice", "is", "person"), ("wonderland", "is", "location")]

        extractor = ModelPreferenceExtractor(
            config,
            tone_model=tone_model,
            emotion_model=emotion_model,
            formality_model=formality_model,
            openie_model=openie_model,
        )
        state = ExtractionState()
        pref = extractor.extract("hi", "this is a short response", state)

        self.assertEqual(pref.tone[0], 1)
        self.assertGreater(pref.length, 0.0)
        self.assertGreater(pref.density, 0.0)
        self.assertGreaterEqual(pref.formality, 0.0)
        self.assertLessEqual(pref.formality, 1.0)


if __name__ == "__main__":
    unittest.main()


