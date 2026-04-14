"""Generate small cropped examples showing what each invented term looks like.

For each of the 14 used invented terms, find the top patches across all images
that most closely match the term's centroid embedding, then crop those patches
(with a small context window) and assemble into a visual glossary.
"""

import json
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
from PIL import Image

from fordera.trait_discovery import extract_patch_embeddings

# CLIP ViT-B/32 uses 32x32 patches on a 224x224 image → 7x7 = 49 patches
PATCH_GRID = 7
PATCH_SIZE = 32  # pixels on 224x224 image
# How much context to show around each patch in the crop
CONTEXT_PAD = 48  # pixels each side (so crop is ~112x112)


def patch_to_bbox(patch_idx: int, img_w: int, img_h: int):
    """Map a patch index (0-48) to a (x1, y1, x2, y2) bounding box on the original image."""
    # Patch grid is 7x7; ViT sees the image at 224x224
    row = patch_idx // PATCH_GRID
    col = patch_idx % PATCH_GRID

    # Scale patch coordinates back to original image dimensions
    px_ratio = PATCH_SIZE / 224.0
    cx = (col + 0.5) * PATCH_SIZE / 224.0 * img_w
    cy = (row + 0.5) * PATCH_SIZE / 224.0 * img_h

    half_w = CONTEXT_PAD * (img_w / 224.0)
    half_h = CONTEXT_PAD * (img_h / 224.0)

    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(img_w, int(cx + half_w))
    y2 = min(img_h, int(cy + half_h))
    return x1, y1, x2, y2


def find_top_patches_per_trait(
    manifest,
    centroids: np.ndarray,
    top_k: int = 4,
    device: str = "cpu",
):
    """For each trait centroid, find the top-k (image_path, patch_idx) pairs
    with highest cosine similarity.
    """
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # (n_images * n_patches, dim)
    all_patches = []
    patch_provenance = []  # (img_idx, patch_idx) for each row
    for img_idx, entry in enumerate(manifest):
        path = Path(entry["processed_path"])
        patches = extract_patch_embeddings(path, clip_model, clip_preprocess, device)
        patches = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
        all_patches.append(patches)
        for p_idx in range(patches.shape[0]):
            patch_provenance.append((img_idx, p_idx))

    X = np.vstack(all_patches)  # (total_patches, dim)

    # For each trait, top-k patches with highest similarity to centroid
    top_per_trait = {}
    for t_idx in range(centroids.shape[0]):
        sims = X @ centroids[t_idx]
        top_indices = np.argsort(-sims)[:top_k]
        top_per_trait[t_idx] = [
            (patch_provenance[i][0], patch_provenance[i][1], float(sims[i]))
            for i in top_indices
        ]
    return top_per_trait


def crop_patch_region(image_path: Path, patch_idx: int, target_size: int = 96) -> np.ndarray:
    """Crop the region around a patch and resize to target_size."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = patch_to_bbox(patch_idx, w, h)
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (target_size, target_size))
    return crop


def build_glossary_image(
    manifest,
    used_traits: list,
    trait_names: dict,
    top_per_trait: dict,
    output_path: Path,
    crop_size: int = 96,
    gutter: int = 8,
    label_height: int = 22,
):
    """Build a single glossary image showing all used terms with their top patches."""
    cols_per_row = 4  # show 4 exemplars per trait
    row_height = label_height + crop_size + gutter
    img_width = cols_per_row * (crop_size + gutter) + gutter + 150  # 150 for label column
    img_height = len(used_traits) * row_height + gutter

    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    for row, t_idx in enumerate(used_traits):
        name = trait_names[t_idx]
        y_top = row * row_height + gutter
        # Draw term name on the left
        cv2.putText(
            canvas,
            name,
            (8, y_top + crop_size // 2 + 4),
            font,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"#{t_idx}",
            (8, y_top + crop_size // 2 + 24),
            font,
            0.4,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        # Draw the top-k patch crops to the right
        for col, (img_idx, patch_idx, score) in enumerate(top_per_trait[t_idx][:cols_per_row]):
            entry = manifest[img_idx]
            path = Path(entry["processed_path"])
            crop = crop_patch_region(path, patch_idx, crop_size)

            x_left = 150 + col * (crop_size + gutter)
            canvas[y_top:y_top + crop_size, x_left:x_left + crop_size] = crop

            # Draw border
            cv2.rectangle(
                canvas,
                (x_left, y_top),
                (x_left + crop_size - 1, y_top + crop_size - 1),
                (180, 180, 180),
                1,
            )

            # Draw the year label below
            year_label = entry["label"]
            cv2.putText(
                canvas,
                year_label,
                (x_left, y_top + crop_size + 15),
                font,
                0.4,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), canvas)
    return output_path


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    out_dir = Path(__file__).parent.parent.parent / "outputs"
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    manifest = json.loads((proc_dir / "manifest.json").read_text())
    tree = json.loads((out_dir / "trait_tree.json").read_text())
    glossary_summary = json.loads((out_dir / "trait_glossary.json").read_text())
    centroids = np.load(out_dir / "trait_centroids.npy")

    # Collect used traits (in tree traversal order)
    used_traits = []
    seen = set()

    def walk(node):
        if node["type"] == "decision":
            if node["trait_id"] not in seen:
                used_traits.append(node["trait_id"])
                seen.add(node["trait_id"])
            walk(node["yes"])
            walk(node["no"])

    walk(tree)
    print(f"Used traits: {used_traits}")

    trait_names = {entry["trait_id"]: entry["name"] for entry in glossary_summary}
    print(f"Computing top patches for {len(used_traits)} traits...")
    top_per_trait = find_top_patches_per_trait(manifest, centroids, top_k=4)

    # Filter to only used traits
    top_used = {t: top_per_trait[t] for t in used_traits}

    out_path = docs_dir / "trait_glossary.png"
    build_glossary_image(manifest, used_traits, trait_names, top_used, out_path)
    print(f"Saved glossary image to {out_path}")
