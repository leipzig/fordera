"""Tests for the preprocessor module."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fordera.preprocessor import (
    TARGET_SIZE,
    detect_year_text,
    mask_year_text,
    preprocess_image,
)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture
def sample_raw_image_path():
    """Return path to a raw image known to have year text."""
    return RAW_DIR / "1952.webp"


@pytest.fixture
def sample_raw_image(sample_raw_image_path):
    """Load a raw image."""
    img = cv2.imread(str(sample_raw_image_path))
    assert img is not None, f"Could not load {sample_raw_image_path}"
    return img


class TestDetectYearText:
    def test_detects_year_in_raw_image(self, sample_raw_image):
        """Raw images should have year text that OCR can detect."""
        regions = detect_year_text(sample_raw_image)
        # At least one region should contain a year
        assert len(regions) > 0, "Expected to find year text in raw image"
        texts = [text for _, text in regions]
        # At least one detected text should contain a year-like string
        found_year = any("195" in t or "194" in t for t in texts)
        assert found_year, f"No year found in detected texts: {texts}"


class TestMaskYearText:
    def test_masked_image_has_no_year_text(self, sample_raw_image):
        """After masking, OCR should not find year text."""
        masked = mask_year_text(sample_raw_image)
        regions = detect_year_text(masked)
        year_texts = [text for _, text in regions]
        assert len(year_texts) == 0, (
            f"Year text still found after masking: {year_texts}"
        )

    def test_masked_image_same_shape(self, sample_raw_image):
        """Masking should not change image dimensions."""
        masked = mask_year_text(sample_raw_image)
        assert masked.shape == sample_raw_image.shape

    def test_non_text_regions_preserved(self, sample_raw_image):
        """Pixels far from text regions should be identical after masking."""
        masked = mask_year_text(sample_raw_image)
        # Compare bottom-right corner (unlikely to have text overlay)
        h, w = sample_raw_image.shape[:2]
        corner_orig = sample_raw_image[h - 50 : h, w - 50 : w]
        corner_masked = masked[h - 50 : h, w - 50 : w]
        # They should be very similar (allow small differences from inpainting bleed)
        diff = np.abs(corner_orig.astype(float) - corner_masked.astype(float))
        assert diff.mean() < 5.0, (
            f"Non-text region changed too much: mean diff = {diff.mean():.1f}"
        )


class TestPreprocessImage:
    def test_output_dimensions(self, sample_raw_image_path):
        """Preprocessed images should match target size."""
        result = preprocess_image(sample_raw_image_path)
        assert result.shape[:2] == (TARGET_SIZE[1], TARGET_SIZE[0])

    def test_output_is_3_channel(self, sample_raw_image_path):
        """Preprocessed images should be 3-channel BGR."""
        result = preprocess_image(sample_raw_image_path)
        assert result.shape[2] == 3

    def test_all_processed_images_exist(self):
        """Every image in the processed manifest should exist on disk."""
        manifest_path = PROCESSED_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Processed manifest not found — run preprocessor first")
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            p = Path(entry["processed_path"])
            assert p.exists(), f"Missing processed image: {p}"

    def test_all_processed_images_correct_size(self):
        """Every processed image should be 224x224."""
        manifest_path = PROCESSED_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Processed manifest not found — run preprocessor first")
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            img = cv2.imread(entry["processed_path"])
            assert img is not None, f"Could not load {entry['processed_path']}"
            assert img.shape[:2] == (TARGET_SIZE[1], TARGET_SIZE[0]), (
                f"{entry['label']}: expected {TARGET_SIZE}, got {img.shape[:2]}"
            )
