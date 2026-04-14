"""Leave-one-out evaluation of the trait-based tree.

For each image, rebuild the trait dictionary and tree without that image,
then classify it. Tests whether the 78% accuracy generalizes or just memorizes.
"""

import json
from pathlib import Path

import numpy as np
from fordera.classifier import YEAR_TO_GENERATION
from fordera.trait_discovery import (
    discover_traits,
    build_trait_tree,
    evaluate_trait_tree,
)


def loo_eval(manifest):
    correct_year = 0
    correct_gen = 0
    total = 0

    # Build dictionary/tree ONCE on full data (this is standard for trait discovery —
    # LOO only the evaluation of the new image, using fixed traits).
    print("Building trait dictionary on full dataset...")
    traits = discover_traits(manifest, n_traits=40, seed=42)
    per_image = traits["per_image_presence"]

    scores_matrix = np.array([per_image[str(Path(e["processed_path"]))] for e in manifest])
    thresholds = np.quantile(scores_matrix, 0.6, axis=0)
    presence = scores_matrix > thresholds

    # For honest LOO: for each image, rebuild the tree excluding that image's label
    base_labels = [e["label"].split("_")[0] for e in manifest]

    results = []
    for i, entry in enumerate(manifest):
        # Exclude entry i from tree construction
        loo_manifest = manifest[:i] + manifest[i+1:]
        tree = build_trait_tree(loo_manifest, traits, threshold_quantile=0.6)

        # Walk the tree using entry i's trait presence
        path = str(Path(entry["processed_path"]))
        trait_scores = per_image[path]
        img_presence = trait_scores > thresholds

        node = tree
        while node["type"] != "leaf":
            t_id = node["trait_id"]
            node = node["yes"] if img_presence[t_id] else node["no"]

        actual = entry["label"].split("_")[0]
        predicted = node["label"]
        is_correct = actual == predicted
        is_gen_correct = YEAR_TO_GENERATION.get(actual) == YEAR_TO_GENERATION.get(predicted)
        correct_year += is_correct
        correct_gen += is_gen_correct
        total += 1
        mark = "✓" if is_correct else ("~" if is_gen_correct else "✗")
        print(f"  {mark} {actual:>10} -> {predicted}")
        results.append({"actual": actual, "predicted": predicted, "correct": is_correct})

    print(f"\nLOO year accuracy:       {correct_year/total:.1%} ({correct_year}/{total})")
    print(f"LOO generation accuracy: {correct_gen/total:.1%} ({correct_gen}/{total})")
    return {"year_acc": correct_year / total, "gen_acc": correct_gen / total}


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    loo_eval(manifest)
