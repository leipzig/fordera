"""Tests for the dichotomous key generator."""

import json
from pathlib import Path

import numpy as np
import pytest

from fordera.keygen import DichotomousKeyGenerator

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
MODEL_DIR = Path(__file__).parent.parent / "models"

ALL_LABELS = {
    "1948-1950", "1951", "1952", "1953", "1954", "1955", "1956",
    "1957", "1958", "1959", "1960", "1961", "1962", "1963", "1964",
    "1965", "1966", "1967", "1968", "1969", "1970", "1971", "1972",
    "1973-1975", "1976-1977", "1978", "1979",
}


@pytest.fixture(scope="module")
def keygen():
    kg = DichotomousKeyGenerator()
    if (MODEL_DIR / "keygen.pkl").exists():
        kg.load(MODEL_DIR)
    else:
        pytest.skip("Key generator model not found")
    return kg


@pytest.fixture(scope="module")
def manifest():
    path = PROCESSED_DIR / "manifest.json"
    if not path.exists():
        pytest.skip("Processed manifest not found")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def key_json(keygen, manifest):
    return keygen.to_interactive_json(manifest)


class TestTreeStructure:
    def test_all_labels_reachable(self, keygen):
        """Every model year in the training set should be reachable."""
        reachable = set(keygen.get_all_labels())
        assert reachable == ALL_LABELS, (
            f"Missing labels: {ALL_LABELS - reachable}"
        )

    def test_tree_is_fitted(self, keygen):
        """The tree should be fitted."""
        assert keygen._is_fitted

    def test_no_orphan_nodes(self, keygen):
        """Every node should be reachable from the root (well-formed tree)."""
        visited_leaves = set()
        visited_decisions = set()

        def _visit(node):
            if node["type"] == "leaf":
                visited_leaves.add(node["label"])
            else:
                visited_decisions.add(node["node_id"])
                _visit(node["left"])
                _visit(node["right"])

        _visit(keygen.tree_root)
        # All labels should be visited
        assert visited_leaves == ALL_LABELS, (
            f"Not all labels visited. Missing: {ALL_LABELS - visited_leaves}"
        )
        # Decision nodes should be fewer than total nodes
        assert len(visited_decisions) > 0, "No decision nodes found"


class TestInteractiveJSON:
    def test_json_root_is_decision(self, key_json):
        """Root node should be a decision node."""
        assert key_json["type"] == "decision"

    def test_decision_nodes_have_question(self, key_json):
        """Every decision node should have a 'question' field."""
        def _check(node):
            if node["type"] == "decision":
                assert "question" in node, "Decision node missing 'question'"
                assert isinstance(node["question"], str)
                assert len(node["question"]) > 0
                _check(node["yes"])
                _check(node["no"])
        _check(key_json)

    def test_leaf_nodes_have_label(self, key_json):
        """Every leaf node should have a valid year label."""
        def _check(node):
            if node["type"] == "leaf":
                assert "label" in node
                assert node["label"] in ALL_LABELS, (
                    f"Leaf label '{node['label']}' not valid"
                )
            else:
                _check(node["yes"])
                _check(node["no"])
        _check(key_json)

    def test_leaf_nodes_have_example_images(self, key_json):
        """Leaf nodes should have example image paths."""
        def _check(node):
            if node["type"] == "leaf":
                assert "example_images" in node
                assert len(node["example_images"]) > 0, (
                    f"No example images for {node['label']}"
                )
                for img_path in node["example_images"]:
                    assert Path(img_path).exists(), (
                        f"Example image not found: {img_path}"
                    )
            else:
                _check(node["yes"])
                _check(node["no"])
        _check(key_json)

    def test_all_labels_in_json(self, key_json):
        """Every label should appear in at least one leaf."""
        found = set()
        def _collect(node):
            if node["type"] == "leaf":
                found.add(node["label"])
            else:
                _collect(node["yes"])
                _collect(node["no"])
        _collect(key_json)
        assert found == ALL_LABELS, f"Missing: {ALL_LABELS - found}"


class TestPrintableOutput:
    def test_svg_exists(self):
        """SVG output should exist."""
        svg_path = OUTPUT_DIR / "dichotomous_key.svg"
        assert svg_path.exists(), f"SVG not found at {svg_path}"
        content = svg_path.read_text()
        assert "<svg" in content, "File does not contain SVG markup"

    def test_pdf_exists(self):
        """PDF output should exist."""
        pdf_path = OUTPUT_DIR / "dichotomous_key.pdf"
        assert pdf_path.exists(), f"PDF not found at {pdf_path}"
        content = pdf_path.read_bytes()
        assert content[:4] == b"%PDF", "File is not a valid PDF"
