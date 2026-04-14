"""Discover visual traits and invent names for them.

Instead of describing features in English, we:
1. Extract patch-level embeddings from every truck illustration
2. Cluster the patches to find recurring visual motifs
3. Invent a phonetic nonsense name for each cluster (e.g., "krozz", "fembrin")
4. Store each name with its cluster centroid embedding
5. Build a dichotomous key where each split asks "Does this truck contain {name}?"
   The answer is determined by visual similarity to the stored embedding,
   not by CLIP's text encoder.

The result: a key whose labels are invented words whose meaning is defined
purely by a visual embedding dictionary.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# Phonetic syllables for inventing trait names. No English words.
ONSETS = ["b", "d", "f", "g", "j", "k", "l", "m", "n", "p", "r", "s", "t",
          "v", "z", "kr", "fr", "gr", "pr", "tr", "br", "dr", "sn", "sl",
          "fl", "gl", "kl", "pl", "th", "sh", "zh", "sk", "sp", "st"]
NUCLEI = ["a", "e", "i", "o", "u", "ae", "ao", "ei", "au", "eu", "y"]
CODAS = ["", "n", "m", "l", "r", "ng", "nt", "lm", "rk", "zz", "x",
         "dd", "ff", "ss", "rl", "rn", "sk", "st", "ch", "lt", "mp"]


def invent_name(rng: random.Random) -> str:
    """Generate a phonetic nonsense name."""
    n_syllables = rng.choice([2, 2, 2, 3])  # usually 2, sometimes 3
    parts = []
    for _ in range(n_syllables):
        parts.append(rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS))
    name = "".join(parts)
    # Keep reasonable length
    return name[:12]


def extract_patch_embeddings(
    image_path: Path, clip_model, clip_preprocess, device: str = "cpu"
) -> np.ndarray:
    """Extract per-patch embeddings from CLIP ViT's intermediate layer.

    Returns array of shape (n_patches, dim).
    """
    img = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Hook into the CLIP ViT to get per-patch tokens before pooling
    tokens = {}
    def hook(module, inputs, output):
        # output shape: (batch, seq_len, dim) for ViT
        tokens["x"] = output

    # CLIP ViT-B/32: the last layer of the visual transformer produces (1, 50, 768):
    # 1 CLS token + 49 patch tokens (7x7 grid)
    handle = clip_model.visual.transformer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = clip_model.encode_image(img)
    finally:
        handle.remove()

    x = tokens["x"]  # (seq_len, batch, dim) for CLIP (transposed)
    # CLIP's transformer returns (seq_len, batch, dim). Transpose to (batch, seq_len, dim)
    if x.dim() == 3 and x.shape[1] == 1:
        x = x.permute(1, 0, 2)
    patches = x.squeeze(0)[1:]  # drop CLS token
    # Project to same space as image embedding so they're comparable
    patches = clip_model.visual.ln_post(patches)
    if clip_model.visual.proj is not None:
        patches = patches @ clip_model.visual.proj
    return patches.detach().cpu().numpy()


def discover_traits(
    manifest: List[dict],
    n_traits: int = 40,
    seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """Cluster patches across all trucks to discover visual traits.

    Returns:
        {
          "names": [...],               # invented names per cluster
          "centroids": np.ndarray,      # (n_traits, dim) centroid embeddings
          "per_image_presence": {path: np.array([...]) of length n_traits},
        }
    """
    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    print(f"Extracting patch embeddings from {len(manifest)} trucks...")
    all_patches = []
    per_image_patches = {}
    for i, entry in enumerate(manifest):
        path = Path(entry["processed_path"])
        patches = extract_patch_embeddings(path, clip_model, clip_preprocess, device)
        # Normalize each patch embedding to unit length for cosine clustering
        patches = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
        per_image_patches[str(path)] = patches
        all_patches.append(patches)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(manifest)}")

    X = np.vstack(all_patches)
    print(f"Clustering {X.shape[0]} patches into {n_traits} trait clusters...")
    kmeans = KMeans(n_clusters=n_traits, random_state=seed, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # Invent names
    rng = random.Random(seed)
    names = []
    name_set = set()
    while len(names) < n_traits:
        n = invent_name(rng)
        if n not in name_set:
            names.append(n)
            name_set.add(n)

    # Compute per-image trait presence: max cosine similarity of any patch to each centroid
    print("Computing per-image trait presence...")
    per_image_presence = {}
    for path, patches in per_image_patches.items():
        # (n_patches, n_traits) = similarity of each patch to each trait centroid
        sims = patches @ centroids.T
        # For each trait, the strongest matching patch in this image
        trait_scores = sims.max(axis=0)
        per_image_presence[path] = trait_scores

    return {
        "names": names,
        "centroids": centroids,
        "per_image_presence": per_image_presence,
    }


def build_trait_tree(
    manifest: List[dict],
    traits: Dict,
    threshold_quantile: float = 0.6,
) -> Dict:
    """Build a dichotomous tree where each split asks about presence of an invented trait.

    At each split, find the trait that most evenly divides the current class set,
    based on whether the trait's max-patch similarity exceeds the per-trait threshold.
    """
    from fordera.classifier import YEAR_TO_GENERATION

    names = traits["names"]
    per_image = traits["per_image_presence"]

    # Compute per-trait presence threshold (quantile across the dataset).
    # A trait is "present" in an image if its score is above this threshold.
    scores_matrix = np.array([per_image[str(Path(e["processed_path"]))] for e in manifest])
    thresholds = np.quantile(scores_matrix, threshold_quantile, axis=0)
    presence = scores_matrix > thresholds  # (n_images, n_traits) boolean

    # Aggregate per base label (average presence across duplicate images)
    base_labels = [e["label"].split("_")[0] for e in manifest]
    unique_labels = sorted(set(base_labels))
    per_label_presence = np.zeros((len(unique_labels), len(names)), dtype=bool)
    for i, ul in enumerate(unique_labels):
        mask = [j for j, bl in enumerate(base_labels) if bl == ul]
        per_label_presence[i] = presence[mask].any(axis=0)

    def build(label_indices, used_traits, depth=0):
        if len(label_indices) == 1:
            return {"type": "leaf", "label": unique_labels[label_indices[0]]}
        if depth > 20:
            # Fallback: just pick the earliest year
            return {"type": "leaf", "label": unique_labels[label_indices[0]]}

        # Find the trait that splits most evenly (closest to 50/50)
        best_trait = None
        best_split_score = -1
        best_yes_indices = None
        best_no_indices = None

        for t_idx in range(len(names)):
            if t_idx in used_traits:
                continue
            yes = [i for i in label_indices if per_label_presence[i, t_idx]]
            no = [i for i in label_indices if not per_label_presence[i, t_idx]]
            if not yes or not no:
                continue
            # Score: prefer balanced splits, tie-break by total label count
            balance = min(len(yes), len(no)) / max(len(yes), len(no))
            if balance > best_split_score:
                best_split_score = balance
                best_trait = t_idx
                best_yes_indices = yes
                best_no_indices = no

        if best_trait is None:
            # No trait separates these labels; emit a leaf with the first
            return {"type": "leaf", "label": unique_labels[label_indices[0]]}

        return {
            "type": "decision",
            "trait_name": names[best_trait],
            "trait_id": int(best_trait),
            "yes": build(best_yes_indices, used_traits | {best_trait}, depth + 1),
            "no": build(best_no_indices, used_traits | {best_trait}, depth + 1),
        }

    return build(list(range(len(unique_labels))), set())


def evaluate_trait_tree(tree: Dict, traits: Dict, manifest: List[dict]) -> Dict:
    """Walk the trait tree for each image using the stored centroid embeddings.

    CLIP patch embeddings are compared against the invented-term centroids
    from the embedding dictionary — no CLIP text encoding involved.
    """
    from fordera.classifier import YEAR_TO_GENERATION

    per_image = traits["per_image_presence"]
    # Compute presence thresholds as in build_trait_tree
    scores_matrix = np.array([per_image[str(Path(e["processed_path"]))] for e in manifest])
    thresholds = np.quantile(scores_matrix, 0.6, axis=0)

    results = []
    for entry in manifest:
        path = str(Path(entry["processed_path"]))
        trait_scores = per_image[path]
        present = trait_scores > thresholds

        node = tree
        while node["type"] != "leaf":
            t_id = node["trait_id"]
            node = node["yes"] if present[t_id] else node["no"]

        actual = entry["label"].split("_")[0]
        predicted = node["label"]
        results.append({
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "gen_correct": YEAR_TO_GENERATION.get(actual) == YEAR_TO_GENERATION.get(predicted),
        })

    total = len(results)
    return {
        "year_acc": sum(r["correct"] for r in results) / total,
        "gen_acc": sum(r["gen_correct"] for r in results) / total,
        "year_correct": sum(r["correct"] for r in results),
        "gen_correct": sum(r["gen_correct"] for r in results),
        "total": total,
        "results": results,
    }


def tree_as_text(tree: Dict, indent: str = "") -> str:
    """Pretty-print the tree."""
    if tree["type"] == "leaf":
        return f"{indent}-> {tree['label']}\n"
    s = f"{indent}Does it have {tree['trait_name']}? (trait #{tree['trait_id']})\n"
    s += f"{indent}  Yes:\n"
    s += tree_as_text(tree["yes"], indent + "    ")
    s += f"{indent}  No:\n"
    s += tree_as_text(tree["no"], indent + "    ")
    return s


def visualize_traits(
    manifest: List[dict],
    traits: Dict,
    output_dir: Path,
    top_k: int = 4,
) -> None:
    """For each invented trait, save the top-k truck illustrations that match
    it most strongly. This is the 'embedding dictionary' rendered visually —
    the meaning of each invented word is defined by the images it matches.
    """
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    names = traits["names"]
    per_image = traits["per_image_presence"]

    scores_matrix = np.array([per_image[str(Path(e["processed_path"]))] for e in manifest])
    paths = [Path(e["processed_path"]) for e in manifest]
    labels = [e["label"] for e in manifest]

    glossary = []
    for t_idx in range(len(names)):
        scores = scores_matrix[:, t_idx]
        top_idx = np.argsort(-scores)[:top_k]

        glossary.append({
            "name": names[t_idx],
            "trait_id": t_idx,
            "top_matches": [
                {"label": labels[i], "path": str(paths[i]), "score": float(scores[i])}
                for i in top_idx
            ],
        })

    (output_dir / "trait_glossary.json").write_text(json.dumps(glossary, indent=2))
    print(f"Glossary saved to {output_dir / 'trait_glossary.json'}")


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    manifest = json.loads((proc_dir / "manifest.json").read_text())

    # Discover traits
    traits = discover_traits(manifest, n_traits=40, seed=42)

    print("\nInvented trait names:")
    print(", ".join(traits["names"]))

    # Build and evaluate tree
    tree = build_trait_tree(manifest, traits, threshold_quantile=0.6)
    print("\nTrait-based dichotomous tree:")
    print(tree_as_text(tree))

    results = evaluate_trait_tree(tree, traits, manifest)
    print(f"\nTrait tree evaluation:")
    print(f"  Year accuracy:       {results['year_acc']:.1%} ({results['year_correct']}/{results['total']})")
    print(f"  Generation accuracy: {results['gen_acc']:.1%} ({results['gen_correct']}/{results['total']})")

    # Save trait dictionary and visualizations
    visualize_traits(manifest, traits, output_dir)

    # Save tree
    (output_dir / "trait_tree.json").write_text(json.dumps(tree, indent=2))
    print(f"Tree saved to {output_dir / 'trait_tree.json'}")

    # Save full traits dict (without embeddings for portability)
    trait_summary = {
        "names": traits["names"],
        "per_image_presence": {k: v.tolist() for k, v in traits["per_image_presence"].items()},
    }
    np.save(output_dir / "trait_centroids.npy", traits["centroids"])
    (output_dir / "trait_summary.json").write_text(json.dumps(trait_summary, indent=2))
    print(f"Centroids saved to {output_dir / 'trait_centroids.npy'}")
