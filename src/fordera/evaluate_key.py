"""Evaluate the dichotomous key as a standalone classifier.

For each image, walk the key tree using CLIP to answer each yes/no
question. No ResNet embeddings or k-NN — just the key and CLIP's
visual understanding.
"""

import json
from pathlib import Path

import clip
import torch
from PIL import Image

from fordera.classifier import YEAR_TO_GENERATION


def evaluate_key_with_clip(key_json: dict, manifest: list) -> dict:
    """Walk the dichotomous key for each image using CLIP to answer questions.

    Returns evaluation results with year-level and generation-level accuracy.
    """
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    def answer_question(image_path: Path, question: str) -> bool:
        """Use CLIP to answer a yes/no question about an image."""
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        # Strip "Does it have " prefix and "?" suffix for cleaner prompts
        feature = question.replace("Does it have ", "").rstrip("?")

        yes_text = f"a pickup truck with {feature}"
        no_text = f"a pickup truck without {feature}"
        tokens = clip.tokenize([yes_text, no_text]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(img, tokens)
            probs = logits_per_image.softmax(dim=-1).squeeze()

        return probs[0].item() > probs[1].item()

    def walk_tree(node: dict, image_path: Path) -> str:
        """Walk the key tree for an image, returning the predicted label."""
        if node["type"] == "leaf":
            return node["label"]

        question = node["question"]
        if answer_question(image_path, question):
            return walk_tree(node["yes"], image_path)
        else:
            return walk_tree(node["no"], image_path)

    results = []
    for entry in manifest:
        path = Path(entry["processed_path"])
        actual = entry["label"].split("_")[0]

        predicted = walk_tree(key_json, path)

        actual_gen = YEAR_TO_GENERATION.get(actual, actual)
        pred_gen = YEAR_TO_GENERATION.get(predicted, predicted)

        results.append({
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "actual_generation": actual_gen,
            "predicted_generation": pred_gen,
            "generation_correct": actual_gen == pred_gen,
        })

    total = len(results)
    year_correct = sum(r["correct"] for r in results)
    gen_correct = sum(r["generation_correct"] for r in results)
    n_classes = len(set(r["actual"] for r in results))
    n_generations = len(set(r["actual_generation"] for r in results))

    return {
        "year_accuracy": year_correct / total,
        "year_correct": year_correct,
        "generation_accuracy": gen_correct / total,
        "generation_correct": gen_correct,
        "total": total,
        "n_classes": n_classes,
        "n_generations": n_generations,
        "year_random_chance": 1.0 / n_classes,
        "generation_random_chance": 1.0 / n_generations,
        "results": results,
    }


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"

    manifest = json.loads((proc_dir / "manifest.json").read_text())
    key_json = json.loads((output_dir / "dichotomous_key.json").read_text())

    print("Evaluating dichotomous key using CLIP only (no classifier)...\n")
    results = evaluate_key_with_clip(key_json, manifest)

    print(f"Year-level accuracy:       {results['year_accuracy']:.1%} "
          f"({results['year_correct']}/{results['total']})")
    print(f"  Random chance:           {results['year_random_chance']:.1%} "
          f"(1/{results['n_classes']})")
    print(f"Generation-level accuracy: {results['generation_accuracy']:.1%} "
          f"({results['generation_correct']}/{results['total']})")
    print(f"  Random chance:           {results['generation_random_chance']:.1%} "
          f"(1/{results['n_generations']})")

    print("\nPer-image results:")
    for r in results["results"]:
        year_mark = "✓" if r["correct"] else "✗"
        gen_mark = "✓" if r["generation_correct"] else ("~" if not r["correct"] else "✓")
        print(f"  {year_mark} {r['actual']:>10} -> {r['predicted']:<10} "
              f"  gen: {gen_mark} {r['actual_generation'][:15]:>15} -> {r['predicted_generation'][:15]}")
