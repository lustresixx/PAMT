"""Comprehensive tests for HierarchicalMemoryTree and related classes."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pamt.config import UpdateConfig
from pamt.core.memory_tree import (
    HierarchicalMemoryTree,
    MemoryNode,
    RetrievalConfig,
)
from pamt.core.types import PreferenceFusion, PreferenceVector


def _make_pref_vector(
    length: float = 0.5,
    density: float = 0.5,
    formality: float = 0.5,
    tone_idx: int = 0,
    emotion_idx: int = 0,
) -> PreferenceVector:
    """Create a PreferenceVector with customizable dimensions."""
    tone_probs = [0.0] * 4
    tone_probs[tone_idx] = 1.0
    emotion_probs = [0.0] * 5
    emotion_probs[emotion_idx] = 1.0
    return PreferenceVector(
        tone=(tone_idx, tone_probs),
        emotion=(emotion_idx, emotion_probs),
        length=length,
        density=density,
        formality=formality,
    )


def _make_mock_embedder(dim: int = 128) -> MagicMock:
    """Create a mock embedder that returns consistent embeddings based on text."""
    mock = MagicMock()

    def embed_fn(text: str) -> list[float]:
        # Generate deterministic embeddings based on text hash.
        h = hash(text) % 1000
        base = [0.0] * dim
        for i in range(min(10, dim)):
            base[(h + i) % dim] = 1.0
        # Normalize
        norm = sum(v * v for v in base) ** 0.5
        if norm > 0:
            base = [v / norm for v in base]
        return base

    mock.embed = embed_fn
    return mock


def _make_mock_llm(category: str = "General", leaf: str = "Task") -> MagicMock:
    """Create a mock LLM that returns a fixed category and leaf label."""
    mock = MagicMock()
    mock.generate.return_value = json.dumps({"category": category, "leaf": leaf})
    return mock


def _build_tree(
    update_config: UpdateConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    embedder: Any = None,
    llm: Any = None,
) -> HierarchicalMemoryTree:
    """Build a HierarchicalMemoryTree with mock dependencies."""
    if update_config is None:
        update_config = UpdateConfig()
    if retrieval_config is None:
        retrieval_config = RetrievalConfig()
    if embedder is None:
        embedder = _make_mock_embedder()
    if llm is None:
        llm = _make_mock_llm()
    return HierarchicalMemoryTree(
        update_config=update_config,
        retrieval_config=retrieval_config,
        embedder=embedder,
        llm=llm,
    )


# =============================================================================
# MemoryNode Tests
# =============================================================================


class TestMemoryNode(unittest.TestCase):
    """Tests for the MemoryNode data class."""

    def test_get_child_returns_none_if_missing(self) -> None:
        node = MemoryNode(name="root")
        self.assertIsNone(node.get_child("nonexistent"))

    def test_get_child_returns_existing(self) -> None:
        node = MemoryNode(name="root")
        child = MemoryNode(name="child")
        node.children["child"] = child
        self.assertIs(node.get_child("child"), child)

    def test_get_or_create_child_creates_new(self) -> None:
        node = MemoryNode(name="root")
        child = node.get_or_create_child("child")
        self.assertEqual(child.name, "child")
        self.assertEqual(child.label, "child")
        self.assertIn("child", node.children)

    def test_get_or_create_child_returns_existing(self) -> None:
        node = MemoryNode(name="root")
        child1 = node.get_or_create_child("child")
        child2 = node.get_or_create_child("child")
        self.assertIs(child1, child2)

    def test_default_fields(self) -> None:
        node = MemoryNode(name="test")
        self.assertEqual(node.name, "test")
        self.assertIsNone(node.embedding)
        self.assertIsNone(node.label)
        self.assertIsNone(node.content_embedding)
        self.assertEqual(node.content_count, 0)
        self.assertEqual(node.recent_texts, [])
        self.assertEqual(node.children, {})


# =============================================================================
# RetrievalConfig Tests
# =============================================================================


class TestRetrievalConfig(unittest.TestCase):
    """Tests for RetrievalConfig defaults and validation."""

    def test_default_values(self) -> None:
        config = RetrievalConfig()
        self.assertEqual(config.similarity_strict, 0.75)
        self.assertEqual(config.similarity_loose, 0.6)
        self.assertEqual(config.max_candidates, 3)
        self.assertEqual(config.max_leaves_per_category, 8)
        self.assertEqual(config.leaf_reuse_threshold, 0.75)
        self.assertEqual(config.leaf_merge_threshold, 0.6)
        self.assertEqual(config.leaf_summary_max_samples, 4)

    def test_custom_values(self) -> None:
        config = RetrievalConfig(similarity_strict=0.9, max_candidates=5)
        self.assertEqual(config.similarity_strict, 0.9)
        self.assertEqual(config.max_candidates, 5)


# =============================================================================
# HierarchicalMemoryTree Initialization Tests
# =============================================================================


class TestTreeInitialization(unittest.TestCase):
    """Tests for HierarchicalMemoryTree initialization."""

    def test_basic_initialization(self) -> None:
        tree = _build_tree()
        self.assertEqual(tree.root.name, "root")
        self.assertEqual(tree.root.label, "root")
        self.assertFalse(tree.root.state.has_data())

    def test_invalid_similarity_config_raises(self) -> None:
        config = RetrievalConfig(similarity_strict=0.5, similarity_loose=0.8)
        with self.assertRaises(ValueError) as ctx:
            _build_tree(retrieval_config=config)
        self.assertIn("similarity_strict must be >= similarity_loose", str(ctx.exception))

    def test_custom_root_name(self) -> None:
        embedder: Any = _make_mock_embedder()
        llm: Any = _make_mock_llm()
        tree = HierarchicalMemoryTree(
            update_config=UpdateConfig(),
            retrieval_config=RetrievalConfig(),
            embedder=embedder,
            llm=llm,
            root_name="custom_root",
        )
        self.assertEqual(tree.root.name, "custom_root")


# =============================================================================
# Update Preference Tests
# =============================================================================


class TestUpdatePreference(unittest.TestCase):
    """Tests for update_preference and update_preference_trace."""

    def test_update_creates_nodes(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.8)
        tree.update_preference(["Learning", "Math"], pref)

        self.assertIn("Learning", tree.root.children)
        category = tree.root.children["Learning"]
        self.assertIn("Math", category.children)
        self.assertTrue(tree.root.state.has_data())
        self.assertTrue(category.state.has_data())
        self.assertTrue(category.children["Math"].state.has_data())

    def test_update_invalid_path_length(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()

        with self.assertRaises(ValueError):
            tree.update_preference(["Single"], pref)

        with self.assertRaises(ValueError):
            tree.update_preference(["A", "B", "C"], pref)

    def test_update_with_content_text(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()
        tree.update_preference(["Cat", "Leaf"], pref, content_text="Hello world")

        leaf = tree.root.children["Cat"].children["Leaf"]
        self.assertIsNotNone(leaf.content_embedding)
        self.assertEqual(leaf.content_count, 1)
        self.assertEqual(leaf.recent_texts, ["Hello world"])

    def test_multiple_updates_accumulate_content(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()
        tree.update_preference(["Cat", "Leaf"], pref, content_text="Text 1")
        tree.update_preference(["Cat", "Leaf"], pref, content_text="Text 2")

        leaf = tree.root.children["Cat"].children["Leaf"]
        self.assertEqual(leaf.content_count, 2)
        self.assertEqual(leaf.recent_texts, ["Text 1", "Text 2"])

    def test_update_preference_trace_returns_weights(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.7)
        trace = tree.update_preference_trace(["A", "Leaf"], pref)

        self.assertEqual(trace["path"], ["A", "Leaf"])
        self.assertEqual(trace["nodes"]["category"], "A")
        self.assertEqual(trace["nodes"]["leaf"], "Leaf")
        self.assertEqual(trace["weights"]["leaf"], 1.0)
        self.assertIn("category", trace["weights"])
        self.assertIn("root", trace["weights"])

    def test_root_weight_scales_with_total_leaves(self) -> None:
        config = UpdateConfig(window_size=1, ema_decay=0.0, fuse_weight=1.0)
        tree = _build_tree(update_config=config)

        tree.update_preference(["A", "a1"], _make_pref_vector(length=1.0))
        tree.update_preference(["B", "b1"], _make_pref_vector(length=0.0))

        root_length_history = tree.root.state.continuous_history["length"]
        self.assertGreaterEqual(len(root_length_history), 2)
        # Second update gets weight=0.5 (1/2 leaves), so blended value is 0.5.
        self.assertAlmostEqual(root_length_history[-1], 0.5, places=5)


# =============================================================================
# Get Preference Tests
# =============================================================================


class TestGetPreference(unittest.TestCase):
    """Tests for get_preference and get_preference_trace."""

    def test_empty_query_returns_root_preference(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.9)
        tree.update_preference(["Cat", "Leaf"], pref)

        result = tree.get_preference("")
        self.assertIsNotNone(result)

    def test_empty_query_trace_shows_strategy(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()
        tree.update_preference(["Cat", "Leaf"], pref)

        result, trace = tree.get_preference_trace("   ")
        self.assertEqual(trace["strategy"], "empty_query")
        self.assertEqual(trace["selected_path"], ["root"])

    def test_get_preference_fallback_on_no_children(self) -> None:
        tree = _build_tree()
        # Tree has no data, should return None.
        result = tree.get_preference("test query")
        self.assertIsNone(result)

    def test_get_preference_trace_with_children(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()
        tree.update_preference(["Math", "Algebra"], pref)

        result, trace = tree.get_preference_trace("math question")
        self.assertIn("steps", trace)
        self.assertIn("strategy", trace)
        self.assertIn("selected_path", trace)


# =============================================================================
# Route Response Tests
# =============================================================================


class TestRouteResponse(unittest.TestCase):
    """Tests for route_response (label inference and leaf selection)."""

    def test_route_response_basic(self) -> None:
        tree = _build_tree(llm=_make_mock_llm(category="Science", leaf="Physics"))
        path = tree.route_response("What is gravity?", "Gravity is a force...")

        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], "Science")
        self.assertEqual(path[1], "Physics")

    def test_route_response_creates_nodes(self) -> None:
        tree = _build_tree(llm=_make_mock_llm(category="Tech", leaf="Python"))
        tree.route_response("How to code?", "Use Python.")

        self.assertIn("Tech", tree.root.children)
        self.assertIn("Python", tree.root.children["Tech"].children)

    def test_route_response_reuses_existing_leaf_before_llm(self) -> None:
        embedder = _make_mock_embedder()
        llm = _make_mock_llm(category="NewCat", leaf="NewLeaf")
        tree = _build_tree(embedder=embedder, llm=llm)
        pref = _make_pref_vector()
        combined_text = HierarchicalMemoryTree._combine_text("continue", "more details")
        tree.update_preference(["Existing", "LeafA"], pref, content_text=combined_text)

        path = tree.route_response("continue", "more details")

        self.assertEqual(path, ["Existing", "LeafA"])
        llm.generate.assert_not_called()

    def test_route_query_is_alias(self) -> None:
        tree = _build_tree(llm=_make_mock_llm(category="Art", leaf="Painting"))
        path1 = tree.route_response("art query")
        path2 = tree.route_query("art query")
        self.assertEqual(path1, path2)


# =============================================================================
# Static Method Tests
# =============================================================================


class TestStaticMethods(unittest.TestCase):
    """Tests for static helper methods."""

    def test_cosine_similarity_identical(self) -> None:
        vec = [1.0, 0.0, 0.0]
        sim = HierarchicalMemoryTree._cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_cosine_similarity_orthogonal(self) -> None:
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = HierarchicalMemoryTree._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0, places=6)

    def test_cosine_similarity_empty(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._cosine_similarity([], []), 0.0)
        self.assertEqual(HierarchicalMemoryTree._cosine_similarity([1.0], []), 0.0)

    def test_cosine_similarity_mismatched_length(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._cosine_similarity([1.0, 2.0], [1.0]), 0.0)

    def test_cosine_similarity_zero_norm(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_extract_json_valid(self) -> None:
        text = 'Here is JSON: {"category": "Math", "leaf": "Algebra"} more text'
        result = HierarchicalMemoryTree._extract_json(text)
        self.assertEqual(result, {"category": "Math", "leaf": "Algebra"})

    def test_extract_json_invalid(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._extract_json("no json here"), {})
        self.assertEqual(HierarchicalMemoryTree._extract_json("{broken}"), {})

    def test_sanitize_label_strips_and_truncates(self) -> None:
        label = HierarchicalMemoryTree._sanitize_label('  "Test Label"  ', "Fallback")
        self.assertEqual(label, "Test Label")

        long_label = "A" * 100
        label = HierarchicalMemoryTree._sanitize_label(long_label, "Fallback")
        self.assertEqual(len(label), 64)

    def test_sanitize_label_fallback(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._sanitize_label("", "Default"), "Default")
        self.assertEqual(HierarchicalMemoryTree._sanitize_label("   ", "Default"), "Default")

    def test_coarsen_leaf_label_splits(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._coarsen_leaf_label("A / B / C"), "A")
        self.assertEqual(HierarchicalMemoryTree._coarsen_leaf_label("X;Y;Z"), "X")
        self.assertEqual(HierarchicalMemoryTree._coarsen_leaf_label("One and Two"), "One")

    def test_coarsen_leaf_label_truncates(self) -> None:
        label = "Word1 Word2 Word3 Word4 Word5 Word6"
        result = HierarchicalMemoryTree._coarsen_leaf_label(label)
        self.assertEqual(result, "Word1 Word2 Word3 Word4")

    def test_combine_text(self) -> None:
        self.assertEqual(
            HierarchicalMemoryTree._combine_text("query", "response"),
            "User: query\nAssistant: response",
        )
        self.assertEqual(HierarchicalMemoryTree._combine_text("query", ""), "User: query")
        self.assertEqual(HierarchicalMemoryTree._combine_text("", "response"), "Assistant: response")
        self.assertEqual(HierarchicalMemoryTree._combine_text("", ""), "")

    def test_fill_prompt(self) -> None:
        template = "Query: {{query}} Response: {{response}} Category: {{category}}"
        filled = HierarchicalMemoryTree._fill_prompt(
            template, query="Q", response="R", category="C"
        )
        self.assertEqual(filled, "Query: Q Response: R Category: C")

    def test_merge_probs(self) -> None:
        probs1 = [0.5, 0.3, 0.2]
        probs2 = [0.1, 0.8, 0.1]
        weights = [0.5, 0.5]
        merged = HierarchicalMemoryTree._merge_probs([probs1, probs2], weights)
        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(sum(merged), 1.0, places=6)

    def test_merge_probs_empty(self) -> None:
        self.assertEqual(HierarchicalMemoryTree._merge_probs([], []), [])


# =============================================================================
# Weighted Merge Tests
# =============================================================================


class TestWeightedMerge(unittest.TestCase):
    """Tests for _weighted_merge of PreferenceFusion objects."""

    def test_weighted_merge_single(self) -> None:
        pref = PreferenceFusion(
            tone=(0, [1.0, 0.0, 0.0, 0.0]),
            emotion=(1, [0.0, 1.0, 0.0, 0.0, 0.0]),
            length=0.8,
            density=0.5,
            formality=0.6,
        )
        merged = HierarchicalMemoryTree._weighted_merge([pref], [1.0])
        self.assertAlmostEqual(merged.length, 0.8, places=6)
        self.assertAlmostEqual(merged.density, 0.5, places=6)
        self.assertAlmostEqual(merged.formality, 0.6, places=6)

    def test_weighted_merge_equal_weights(self) -> None:
        pref1 = PreferenceFusion(
            tone=(0, [1.0, 0.0, 0.0, 0.0]),
            emotion=(0, [1.0, 0.0, 0.0, 0.0, 0.0]),
            length=0.2,
            density=0.4,
            formality=0.6,
        )
        pref2 = PreferenceFusion(
            tone=(1, [0.0, 1.0, 0.0, 0.0]),
            emotion=(1, [0.0, 1.0, 0.0, 0.0, 0.0]),
            length=0.8,
            density=0.6,
            formality=0.4,
        )
        merged = HierarchicalMemoryTree._weighted_merge([pref1, pref2], [1.0, 1.0])
        self.assertAlmostEqual(merged.length, 0.5, places=6)
        self.assertAlmostEqual(merged.density, 0.5, places=6)
        self.assertAlmostEqual(merged.formality, 0.5, places=6)

    def test_weighted_merge_zero_weights_uses_uniform(self) -> None:
        pref1 = PreferenceFusion(
            tone=(0, [1.0, 0.0, 0.0, 0.0]),
            emotion=(0, [1.0, 0.0, 0.0, 0.0, 0.0]),
            length=0.0,
            density=0.0,
            formality=0.0,
        )
        pref2 = PreferenceFusion(
            tone=(1, [0.0, 1.0, 0.0, 0.0]),
            emotion=(1, [0.0, 1.0, 0.0, 0.0, 0.0]),
            length=1.0,
            density=1.0,
            formality=1.0,
        )
        merged = HierarchicalMemoryTree._weighted_merge([pref1, pref2], [0.0, 0.0])
        self.assertAlmostEqual(merged.length, 0.5, places=6)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization(unittest.TestCase):
    """Tests for to_dict, load_from_dict, save, and load."""

    def test_to_dict_and_load_from_dict(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.7, density=0.3)
        tree.update_preference(["Cat1", "Leaf1"], pref, content_text="Sample text")

        data = tree.to_dict()
        self.assertIn("root_name", data)
        self.assertIn("root", data)

        new_tree = _build_tree()
        new_tree.load_from_dict(data)

        self.assertIn("Cat1", new_tree.root.children)
        self.assertIn("Leaf1", new_tree.root.children["Cat1"].children)
        self.assertTrue(new_tree.root.state.has_data())

    def test_save_and_load(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.5)
        tree.update_preference(["Tech", "Python"], pref, content_text="Python is great")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "tree.json"
            tree.save(path)
            self.assertTrue(path.is_file())

            new_tree = _build_tree()
            new_tree.load(path)
            self.assertIn("Tech", new_tree.root.children)
            leaf = new_tree.root.children["Tech"].children["Python"]
            self.assertEqual(leaf.recent_texts, ["Python is great"])

    def test_load_nonexistent_file_raises(self) -> None:
        tree = _build_tree()
        with self.assertRaises(FileNotFoundError):
            tree.load("/nonexistent/path/file.json")

    def test_load_invalid_json_raises(self) -> None:
        tree = _build_tree()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("[1, 2, 3]")  # Array, not object
            path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                tree.load(path)
            self.assertIn("must contain a JSON object", str(ctx.exception))
        finally:
            Path(path).unlink()


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshot(unittest.TestCase):
    """Tests for the snapshot method."""

    def test_snapshot_empty_tree(self) -> None:
        tree = _build_tree()
        snap = tree.snapshot()
        self.assertEqual(snap["name"], "root")
        self.assertEqual(snap["path"], ["root"])
        self.assertFalse(snap["has_data"])
        self.assertEqual(snap["children"], [])

    def test_snapshot_with_data(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector()
        tree.update_preference(["A", "B"], pref)

        snap = tree.snapshot()
        self.assertTrue(snap["has_data"])
        self.assertEqual(len(snap["children"]), 1)
        self.assertEqual(snap["children"][0]["name"], "A")


# =============================================================================
# Partition Candidates Tests
# =============================================================================


class TestPartitionCandidates(unittest.TestCase):
    """Tests for _partition_candidates."""

    def test_partition_by_score(self) -> None:
        tree = _build_tree(
            retrieval_config=RetrievalConfig(
                similarity_strict=0.8, similarity_loose=0.5, max_candidates=2
            )
        )
        node1 = MemoryNode(name="high")
        node2 = MemoryNode(name="medium")
        node3 = MemoryNode(name="low")
        scored = [(0.9, node1), (0.6, node2), (0.3, node3)]

        strong, weak = tree._partition_candidates(scored)
        self.assertEqual(len(strong), 1)
        self.assertEqual(strong[0][1].name, "high")
        self.assertEqual(len(weak), 1)
        self.assertEqual(weak[0][1].name, "medium")


# =============================================================================
# Format Samples Tests
# =============================================================================


class TestFormatSamples(unittest.TestCase):
    """Tests for _format_samples."""

    def test_format_samples_empty(self) -> None:
        tree = _build_tree()
        self.assertEqual(tree._format_samples([]), "None")

    def test_format_samples_list(self) -> None:
        tree = _build_tree()
        result = tree._format_samples(["Sample A", "Sample B"])
        self.assertIn("1. Sample A", result)
        self.assertIn("2. Sample B", result)

    def test_format_samples_truncates(self) -> None:
        tree = _build_tree(
            retrieval_config=RetrievalConfig(leaf_summary_max_samples=2)
        )
        result = tree._format_samples(["A", "B", "C", "D"])
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("C", result)
        self.assertIn("D", result)


# =============================================================================
# Content Embedding Update Tests
# =============================================================================


class TestUpdateLeafContent(unittest.TestCase):
    """Tests for _update_leaf_content."""

    def test_update_leaf_content_first_time(self) -> None:
        tree = _build_tree()
        leaf = MemoryNode(name="leaf")
        tree._update_leaf_content(leaf, "Hello world")

        self.assertIsNotNone(leaf.content_embedding)
        self.assertEqual(leaf.content_count, 1)
        self.assertEqual(leaf.recent_texts, ["Hello world"])

    def test_update_leaf_content_accumulates(self) -> None:
        tree = _build_tree()
        leaf = MemoryNode(name="leaf")
        tree._update_leaf_content(leaf, "Text 1")
        tree._update_leaf_content(leaf, "Text 2")
        tree._update_leaf_content(leaf, "Text 3")

        self.assertEqual(leaf.content_count, 3)
        self.assertEqual(len(leaf.recent_texts), 3)

    def test_update_leaf_content_empty_text_ignored(self) -> None:
        tree = _build_tree()
        leaf = MemoryNode(name="leaf")
        tree._update_leaf_content(leaf, "")
        tree._update_leaf_content(leaf, "   ")
        tree._update_leaf_content(leaf, None)

        self.assertIsNone(leaf.content_embedding)
        self.assertEqual(leaf.content_count, 0)

    def test_recent_texts_truncated(self) -> None:
        tree = _build_tree(
            retrieval_config=RetrievalConfig(leaf_summary_max_samples=3)
        )
        leaf = MemoryNode(name="leaf")
        for i in range(5):
            tree._update_leaf_content(leaf, f"Text {i}")

        self.assertEqual(len(leaf.recent_texts), 3)
        self.assertEqual(leaf.recent_texts, ["Text 2", "Text 3", "Text 4"])


# =============================================================================
# State Serialization Tests
# =============================================================================


class TestStateSerialization(unittest.TestCase):
    """Tests for _state_to_dict and _state_from_dict."""

    def test_state_roundtrip(self) -> None:
        tree = _build_tree()
        pref = _make_pref_vector(length=0.6)
        tree.update_preference(["Cat", "Leaf"], pref)

        original_state = tree.root.state
        state_dict = tree._state_to_dict(original_state)
        restored_state = tree._state_from_dict(state_dict)

        self.assertEqual(
            original_state.continuous_history, restored_state.continuous_history
        )
        self.assertEqual(
            original_state.categorical_history, restored_state.categorical_history
        )

    def test_state_from_dict_handles_invalid_input(self) -> None:
        state = HierarchicalMemoryTree._state_from_dict(None)  # type: ignore[arg-type]
        self.assertFalse(state.has_data())

        state = HierarchicalMemoryTree._state_from_dict("invalid")  # type: ignore[arg-type]
        self.assertFalse(state.has_data())


# =============================================================================
# Fusion Payload Tests
# =============================================================================


class TestFusionPayload(unittest.TestCase):
    """Tests for _fusion_payload."""

    def test_fusion_payload_none(self) -> None:
        self.assertIsNone(HierarchicalMemoryTree._fusion_payload(None))

    def test_fusion_payload_structure(self) -> None:
        fusion = PreferenceFusion(
            tone=(1, [0.2, 0.8, 0.0, 0.0]),
            emotion=(2, [0.1, 0.1, 0.8, 0.0, 0.0]),
            length=0.5,
            density=0.6,
            formality=0.7,
        )
        payload = HierarchicalMemoryTree._fusion_payload(fusion)

        self.assertEqual(payload["length"], 0.5)
        self.assertEqual(payload["density"], 0.6)
        self.assertEqual(payload["formality"], 0.7)
        self.assertEqual(payload["tone"]["index"], 1)
        self.assertEqual(payload["emotion"]["index"], 2)


if __name__ == "__main__":
    unittest.main()
