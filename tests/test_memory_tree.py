import unittest

from pamt.config import EmbeddingConfig, ModelConfig, UpdateConfig
from pamt.core.memory_tree import HierarchicalMemoryTree, RetrievalConfig
from pamt.core.types import PreferenceVector
from pamt.embeddings.models import OllamaEmbeddings
from pamt.llms.models import OllamaLLM


def _pref_vector(length: float) -> PreferenceVector:
    return PreferenceVector(
        tone=(0, [1.0, 0.0, 0.0, 0.0]),
        emotion=(0, [1.0, 0.0, 0.0, 0.0, 0.0]),
        length=length,
        density=0.5,
        formality=0.5,
    )


def _build_tree(update_config: UpdateConfig) -> HierarchicalMemoryTree:
    embedder = OllamaEmbeddings(EmbeddingConfig())
    llm = OllamaLLM(ModelConfig())
    retrieval = RetrievalConfig()
    return HierarchicalMemoryTree(
        update_config=update_config,
        retrieval_config=retrieval,
        embedder=embedder,
        llm=llm,
        label_prompt_path="prompts/category_leaf.txt",
    )


class MemoryTreeTests(unittest.TestCase):
    def test_update_creates_nodes(self) -> None:
        tree = _build_tree(UpdateConfig())
        pref = _pref_vector(0.8)
        tree.update_preference(["Learning", "Math"], pref)

        self.assertIn("Learning", tree.root.children)
        self.assertIn("Math", tree.root.children["Learning"].children)
        self.assertTrue(tree.root.state.has_data())
        self.assertTrue(tree.root.children["Learning"].state.has_data())
        self.assertTrue(tree.root.children["Learning"].children["Math"].state.has_data())

    def test_root_weight_scales_update(self) -> None:
        config = UpdateConfig(window_size=1, ema_decay=0.0, fuse_weight=1.0)
        tree = _build_tree(config)

        tree.update_preference(["A", "a1"], _pref_vector(1.0))
        tree.update_preference(["B", "b1"], _pref_vector(0.0))

        root_length_history = tree.root.state.continuous_history["length"]
        self.assertGreaterEqual(len(root_length_history), 2)
        self.assertAlmostEqual(root_length_history[-1], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()


