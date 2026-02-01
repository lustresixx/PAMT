import unittest

from pamt.config import PreferenceConfig
from pamt.extractors.preference_models import _go_emotion_scores_to_distribution, _irony_scores_to_tone


class PreferenceModelMappingTests(unittest.TestCase):
    def test_irony_scores_map_to_casual(self) -> None:
        config = PreferenceConfig()
        scores = [
            {"label": "irony", "score": 0.9},
            {"label": "non_irony", "score": 0.1},
        ]
        idx, probs = _irony_scores_to_tone(scores, config.tone_labels)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertEqual(idx, config.tone_labels.index("casual"))

    def test_non_irony_scores_map_to_formal(self) -> None:
        config = PreferenceConfig()
        scores = [{"label": "non_irony", "score": 0.8}]
        idx, _probs = _irony_scores_to_tone(scores, config.tone_labels)
        self.assertEqual(idx, config.tone_labels.index("formal"))

    def test_go_emotions_map_to_base_labels(self) -> None:
        config = PreferenceConfig()
        scores = [
            {"label": "gratitude", "score": 0.7},
            {"label": "sadness", "score": 0.3},
        ]
        idx, probs = _go_emotion_scores_to_distribution(scores, config.emotion_labels)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertEqual(idx, config.emotion_labels.index("joy"))


if __name__ == "__main__":
    unittest.main()


