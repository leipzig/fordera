"""Preprocess truck images: detect and mask year text, resize for model input."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image

# Target size for model input (ResNet/EfficientNet standard)
TARGET_SIZE = (224, 224)

# Regex to match year-like strings (1948-1979 range, also multi-year like "1973-1975")
YEAR_PATTERN = re.compile(r"\b(19[4-7]\d)\b")


def _get_reader() -> easyocr.Reader:
    """Lazy-init EasyOCR reader."""
    if not hasattr(_get_reader, "_reader"):
        _get_reader._reader = easyocr.Reader(["en"], gpu=False)
    return _get_reader._reader


def detect_year_text(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Detect text regions containing year numbers.

    Returns list of (bounding_box, text) tuples where bounding_box is a 4x2 array.
    """
    reader = _get_reader()
    results = reader.readtext(image)

    year_regions = []
    for bbox, text, _conf in results:
        # Check if the detected text contains a year in our range
        if YEAR_PATTERN.search(text):
            year_regions.append((np.array(bbox, dtype=np.int32), text))
    return year_regions


def mask_year_text(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """Detect year text in image and inpaint those regions.

    Args:
        image: BGR image as numpy array
        padding: Extra pixels around detected text to mask

    Returns:
        Image with year text inpainted
    """
    regions = detect_year_text(image)
    if not regions:
        return image.copy()

    # Create binary mask for inpainting
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for bbox, _text in regions:
        # Expand bounding box by padding
        pts = bbox.reshape((-1, 1, 2))
        # Get bounding rect and expand
        x, y, w, h = cv2.boundingRect(pts)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        mask[y : y + h, x : x + w] = 255

    # Inpaint using Telea method
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return result


def preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = TARGET_SIZE,
    mask_text: bool = True,
) -> np.ndarray:
    """Load, mask text, and resize an image for model input.

    Args:
        image_path: Path to the input image
        target_size: (width, height) to resize to
        mask_text: Whether to detect and mask year text

    Returns:
        Preprocessed BGR image as numpy array
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if mask_text:
        image = mask_year_text(image)

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def preprocess_dataset(
    manifest_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> List[dict]:
    """Preprocess all images in the manifest.

    Returns updated manifest with processed image paths.
    """
    import json

    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_manifest = []

    for entry in manifest:
        src = Path(entry["path"])
        dst = output_dir / f"{entry['label']}.png"

        print(f"Processing {entry['label']}...")
        processed = preprocess_image(src, target_size)
        cv2.imwrite(str(dst), processed)

        processed_manifest.append({
            **entry,
            "processed_path": str(dst),
        })

    # Save processed manifest
    out_manifest = output_dir / "manifest.json"
    out_manifest.write_text(json.dumps(processed_manifest, indent=2))
    print(f"Processed manifest saved to {out_manifest}")
    return processed_manifest


if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    preprocess_dataset(raw_dir / "manifest.json", proc_dir)
