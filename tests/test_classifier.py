"""Tests for the classifier module."""

import json
from pathlib import Path

import numpy as np
import pytest

from fordera.classifier import TruckClassifier, FeatureExtractor, YEAR_TO_GENERATION

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

VALID_YEARS = {
    "1948-1950", "1951", "1952", "1953", "1954", "1955", "1956",
    "1957", "1958", "1959", "1960", "1961", "1962", "1963", "1964",
    "1965", "1966", "1967", "1968", "1969", "1970", "1971", "1972",
    "1973-1975", "1976-1977", "1978", "1979",
}


@pytest.fixture(scope="module")
def manifest():
    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Processed manifest not found")
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="module")
def trained_classifier():
    clf = TruckClassifier()
    if (MODEL_DIR / "classifier.pkl").exists():
        clf.load(MODEL_DIR)
    else:
        pytest.skip("Trained model not found")
    return clf


class TestFeatureExtractor:
    def test_embedding_shape(self, manifest):
        """Feature vectors should be 2048-dimensional (ResNet50)."""
        extractor = FeatureExtractor()
        path = Path(manifest[0]["processed_path"])
        features = extractor.extract(path)
        assert features.shape == (2048,), f"Expected (2048,), got {features.shape}"

    def test_different_images_different_embeddings(self, manifest):
        """Different images should produce different embeddings."""
        extractor = FeatureExtractor()
        feat1 = extractor.extract(Path(manifest[0]["processed_path"]))
        feat2 = extractor.extract(Path(manifest[5]["processed_path"]))
        assert not np.allclose(feat1, feat2), "Different images produced identical embeddings"


class TestClassifierPredictions:
    def test_prediction_is_valid_year(self, trained_classifier, manifest):
        """Predictions should be valid year labels."""
        path = Path(manifest[0]["processed_path"])
        label, conf, probs = trained_classifier.predict(path)
        assert label in VALID_YEARS, f"Predicted '{label}' not in valid years"

    def test_confidence_is_probability(self, trained_classifier, manifest):
        """Confidence should be between 0 and 1."""
        path = Path(manifest[0]["processed_path"])
        label, conf, probs = trained_classifier.predict(path)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range"

    def test_probabilities_sum_to_one(self, trained_classifier, manifest):
        """All class probabilities should sum to ~1."""
        path = Path(manifest[0]["processed_path"])
        label, conf, probs = trained_classifier.predict(path)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, not 1.0"

    def test_probabilities_cover_all_classes(self, trained_classifier, manifest):
        """Probability dict should have an entry for every class."""
        path = Path(manifest[0]["processed_path"])
        label, conf, probs = trained_classifier.predict(path)
        assert set(probs.keys()) == VALID_YEARS

    def test_handles_varying_input_sizes(self, trained_classifier):
        """Classifier should handle images of different sizes via transforms."""
        # Create a small test image
        from PIL import Image
        import tempfile
        for size in [(100, 100), (500, 300), (224, 224)]:
            img = Image.new("RGB", size, color="red")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
                label, conf, probs = trained_classifier.predict(Path(f.name))
                assert label in VALID_YEARS


class TestClassifierAccuracy:
    def test_generation_accuracy_above_chance(self, manifest):
        """LOO generation-level accuracy should exceed random chance.

        Exact year accuracy is expected to be low since many years within
        a generation have nearly identical front-end styling. Generation
        accuracy is the meaningful metric for this dataset.
        """
        clf = TruckClassifier()
        results = clf.evaluate_loo(manifest)
        print(f"\nLOO year accuracy: {results['accuracy']:.1%} "
              f"({results['correct']}/{results['total']})")
        print(f"LOO generation accuracy: {results['generation_accuracy']:.1%} "
              f"({results['generation_correct']}/{results['total']})")
        print(f"Generation random chance: {results['generation_random_chance']:.1%} "
              f"(1/{results['n_generations']})")
        for r in results["results"]:
            gen_match = YEAR_TO_GENERATION.get(r["actual"]) == YEAR_TO_GENERATION.get(r["predicted"])
            status = "✓" if r["correct"] else ("~" if gen_match else "✗")
            print(f"  {status} {r['actual']} -> {r['predicted']}")
        assert results["generation_above_chance"], (
            f"Generation accuracy {results['generation_accuracy']:.1%} not above "
            f"chance {results['generation_random_chance']:.1%}"
        )
