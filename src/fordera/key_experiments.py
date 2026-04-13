"""Iterate on dichotomous key strategies to improve text-based classification.

Each experiment generates a key variant and evaluates it with CLIP.
"""

import json
from pathlib import Path
from typing import List

import clip
import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.preprocessing import normalize

from fordera.classifier import FeatureExtractor, YEAR_TO_GENERATION
from fordera.describer import (
    CLIPDescriber,
    FEATURE_VOCABULARY,
    crop_region_for_question,
)


def load_data():
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    manifest = json.loads((proc_dir / "manifest.json").read_text())
    extractor = FeatureExtractor()
    embeddings = []
    labels = []
    for entry in manifest:
        feat = extractor.extract(Path(entry["processed_path"]))
        embeddings.append(feat)
        labels.append(entry["label"])
    return manifest, np.array(embeddings), labels


def build_tree(embeddings, labels, method="ward"):
    """Build a hierarchical clustering tree."""
    base_labels = [l.split("_")[0] for l in labels]
    unique_labels = sorted(set(base_labels))
    avg_embeddings = []
    for ul in unique_labels:
        mask = [i for i, bl in enumerate(base_labels) if bl == ul]
        avg_embeddings.append(embeddings[mask].mean(axis=0))
    avg_embeddings = normalize(np.array(avg_embeddings))

    Z = linkage(avg_embeddings, method=method)
    root = to_tree(Z)

    def min_year(node):
        if node.is_leaf():
            return int(unique_labels[node.id].split("-")[0])
        return min(min_year(node.get_left()), min_year(node.get_right()))

    def reorder(node):
        if node.is_leaf():
            return
        reorder(node.get_left())
        reorder(node.get_right())
        if min_year(node.get_right()) < min_year(node.get_left()):
            node.left, node.right = node.right, node.left

    reorder(root)
    return root, unique_labels


def collect_leaves(node, unique_labels):
    if node.is_leaf():
        return [unique_labels[node.id]]
    return collect_leaves(node.get_left(), unique_labels) + collect_leaves(
        node.get_right(), unique_labels
    )


def evaluate_key(key_tree, manifest, answer_fn):
    """Evaluate a key tree using a given answer function.

    answer_fn(image_path, question) -> bool
    """
    results = []
    for entry in manifest:
        path = Path(entry["processed_path"])
        actual = entry["label"].split("_")[0]

        node = key_tree
        while node["type"] != "leaf":
            answer = answer_fn(path, node["question"])
            node = node["yes"] if answer else node["no"]

        predicted = node["label"]
        actual_gen = YEAR_TO_GENERATION.get(actual, actual)
        pred_gen = YEAR_TO_GENERATION.get(predicted, predicted)

        results.append({
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "gen_correct": actual_gen == pred_gen,
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


# ---- Answer functions ----

def make_clip_answerer(model, preprocess, device="cpu"):
    """Standard CLIP answerer: 'a pickup truck with X' vs 'without X'."""
    def answer(image_path, question):
        feature = question.replace("Does it have ", "").rstrip("?")
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        yes_text = f"a pickup truck with {feature}"
        no_text = f"a pickup truck without {feature}"
        tokens = clip.tokenize([yes_text, no_text]).to(device)
        with torch.no_grad():
            logits, _ = model(img, tokens)
            probs = logits.softmax(dim=-1).squeeze()
        return probs[0].item() > probs[1].item()
    return answer


def make_cropped_clip_answerer(model, preprocess, device="cpu"):
    """CLIP answerer that crops to the region of interest before answering."""
    import cv2

    def answer(image_path, question):
        feature = question.replace("Does it have ", "").rstrip("?")
        crop = crop_region_for_question(question)

        # Load and crop
        raw = cv2.imread(str(image_path))
        h, w = raw.shape[:2]
        y1, y2, x1, x2 = crop
        cropped = raw[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(cropped_rgb)

        img = preprocess(pil).unsqueeze(0).to(device)
        yes_text = f"a pickup truck with {feature}"
        no_text = f"a pickup truck without {feature}"
        tokens = clip.tokenize([yes_text, no_text]).to(device)
        with torch.no_grad():
            logits, _ = model(img, tokens)
            probs = logits.softmax(dim=-1).squeeze()
        return probs[0].item() > probs[1].item()
    return answer


def make_detailed_clip_answerer(model, preprocess, device="cpu"):
    """CLIP answerer with more detailed prompts."""
    def answer(image_path, question):
        feature = question.replace("Does it have ", "").rstrip("?")
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        yes_text = f"a front view illustration of a classic Ford pickup truck that has {feature}"
        no_text = f"a front view illustration of a classic Ford pickup truck that does not have {feature}"
        tokens = clip.tokenize([yes_text, no_text], truncate=True).to(device)
        with torch.no_grad():
            logits, _ = model(img, tokens)
            probs = logits.softmax(dim=-1).squeeze()
        return probs[0].item() > probs[1].item()
    return answer


def make_cropped_detailed_clip_answerer(model, preprocess, device="cpu"):
    """Combines cropping + detailed prompts."""
    import cv2

    def answer(image_path, question):
        feature = question.replace("Does it have ", "").rstrip("?")
        crop = crop_region_for_question(question)

        raw = cv2.imread(str(image_path))
        h, w = raw.shape[:2]
        y1, y2, x1, x2 = crop
        cropped = raw[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(cropped_rgb)

        img = preprocess(pil).unsqueeze(0).to(device)
        yes_text = f"a close-up of a classic Ford pickup truck showing {feature}"
        no_text = f"a close-up of a classic Ford pickup truck not showing {feature}"
        tokens = clip.tokenize([yes_text, no_text], truncate=True).to(device)
        with torch.no_grad():
            logits, _ = model(img, tokens)
            probs = logits.softmax(dim=-1).squeeze()
        return probs[0].item() > probs[1].item()
    return answer


# ---- Key construction variants ----

def build_standard_key(root, unique_labels, manifest, describer, excluded=None):
    """Standard key: CLIP picks best distinguishing feature, no repeats from ancestors."""
    if excluded is None:
        excluded = set()

    label_to_paths = {}
    for entry in manifest:
        bl = entry["label"].split("_")[0]
        if bl not in label_to_paths:
            label_to_paths[bl] = []
        label_to_paths[bl].append(Path(entry["processed_path"]))

    def build(node, excl):
        if node.is_leaf():
            label = unique_labels[node.id]
            return {"type": "leaf", "label": label}

        left_labels = collect_leaves(node.get_left(), unique_labels)
        right_labels = collect_leaves(node.get_right(), unique_labels)

        left_paths = [p for l in left_labels for p in label_to_paths.get(l, [])]
        right_paths = [p for l in right_labels for p in label_to_paths.get(l, [])]

        question, desc, score = describer.best_distinguishing_feature(
            left_paths[:10], right_paths[:10], excluded_questions=excl
        )

        return {
            "type": "decision",
            "question": question,
            "yes": build(node.get_left(), excl | {question}),
            "no": build(node.get_right(), excl | {question}),
        }

    return build(root, excluded)


def build_generation_key(root, unique_labels, manifest, describer):
    """Two-level key: first split to generation, then within-generation splits."""
    # Same as standard but allows question reuse across different subtrees
    # (only ancestors are excluded, not siblings)
    return build_standard_key(root, unique_labels, manifest, describer)


if __name__ == "__main__":
    print("Loading data...")
    manifest, embeddings, labels = load_data()

    print("Loading CLIP...")
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    describer = CLIPDescriber()

    # Build tree
    print("Building tree...")
    root, unique_labels = build_tree(embeddings, labels, method="ward")

    # Build keys
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline — standard key + standard CLIP answerer")
    print("=" * 70)
    key1 = build_standard_key(root, unique_labels, manifest, describer)
    ans1 = make_clip_answerer(model, preprocess, device)
    res1 = evaluate_key(key1, manifest, ans1)
    print(f"  Year accuracy:       {res1['year_acc']:.1%} ({res1['year_correct']}/{res1['total']})")
    print(f"  Generation accuracy: {res1['gen_acc']:.1%} ({res1['gen_correct']}/{res1['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cropped images — crop to region of interest before asking")
    print("=" * 70)
    ans2 = make_cropped_clip_answerer(model, preprocess, device)
    res2 = evaluate_key(key1, manifest, ans2)
    print(f"  Year accuracy:       {res2['year_acc']:.1%} ({res2['year_correct']}/{res2['total']})")
    print(f"  Generation accuracy: {res2['gen_acc']:.1%} ({res2['gen_correct']}/{res2['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Detailed prompts — more specific CLIP prompts")
    print("=" * 70)
    ans3 = make_detailed_clip_answerer(model, preprocess, device)
    res3 = evaluate_key(key1, manifest, ans3)
    print(f"  Year accuracy:       {res3['year_acc']:.1%} ({res3['year_correct']}/{res3['total']})")
    print(f"  Generation accuracy: {res3['gen_acc']:.1%} ({res3['gen_correct']}/{res3['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Cropped + detailed prompts combined")
    print("=" * 70)
    ans4 = make_cropped_detailed_clip_answerer(model, preprocess, device)
    res4 = evaluate_key(key1, manifest, ans4)
    print(f"  Year accuracy:       {res4['year_acc']:.1%} ({res4['year_correct']}/{res4['total']})")
    print(f"  Generation accuracy: {res4['gen_acc']:.1%} ({res4['gen_correct']}/{res4['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Different clustering — complete linkage")
    print("=" * 70)
    root5, ul5 = build_tree(embeddings, labels, method="complete")
    key5 = build_standard_key(root5, ul5, manifest, describer)
    res5 = evaluate_key(key5, manifest, ans1)
    print(f"  Year accuracy:       {res5['year_acc']:.1%} ({res5['year_correct']}/{res5['total']})")
    print(f"  Generation accuracy: {res5['gen_acc']:.1%} ({res5['gen_correct']}/{res5['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Different clustering — average linkage")
    print("=" * 70)
    root6, ul6 = build_tree(embeddings, labels, method="average")
    key6 = build_standard_key(root6, ul6, manifest, describer)
    res6 = evaluate_key(key6, manifest, ans1)
    print(f"  Year accuracy:       {res6['year_acc']:.1%} ({res6['year_correct']}/{res6['total']})")
    print(f"  Generation accuracy: {res6['gen_acc']:.1%} ({res6['gen_correct']}/{res6['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Random forest of keys — vote across multiple diverse trees")
    print("=" * 70)

    # Build a forest: multiple trees using different clustering methods + bootstrap samples
    print("  Building forest (this takes a while)...")
    forest = []
    methods = ["ward", "complete", "average", "ward", "complete"]
    rng = np.random.RandomState(42)

    for i, method in enumerate(methods):
        # Bootstrap: resample the per-label averaged embeddings with replacement
        # to induce diversity in tree structure
        base_labels = [l.split("_")[0] for l in labels]
        unique_labels = sorted(set(base_labels))
        avg_embs = []
        for ul in unique_labels:
            mask = [j for j, bl in enumerate(base_labels) if bl == ul]
            # Bootstrap: sample with replacement from this label's images
            sampled = rng.choice(mask, size=len(mask), replace=True)
            avg_embs.append(embeddings[sampled].mean(axis=0))
        avg_embs = normalize(np.array(avg_embs))

        Z = linkage(avg_embs, method=method)
        root_i = to_tree(Z)

        def min_year(node):
            if node.is_leaf():
                return int(unique_labels[node.id].split("-")[0])
            return min(min_year(node.get_left()), min_year(node.get_right()))

        def reorder(node):
            if node.is_leaf():
                return
            reorder(node.get_left())
            reorder(node.get_right())
            if min_year(node.get_right()) < min_year(node.get_left()):
                node.left, node.right = node.right, node.left

        reorder(root_i)
        key_i = build_standard_key(root_i, unique_labels, manifest, describer)
        forest.append(key_i)
        print(f"    Tree {i+1}/{len(methods)} built ({method})")

    # Evaluate by walking each tree and majority-voting
    from collections import Counter

    forest_results = []
    for entry in manifest:
        path = Path(entry["processed_path"])
        actual = entry["label"].split("_")[0]

        votes = []
        for tree in forest:
            node = tree
            while node["type"] != "leaf":
                answer = ans1(path, node["question"])
                node = node["yes"] if answer else node["no"]
            votes.append(node["label"])

        # Majority vote; fallback to most common generation if year vote tied
        vote_counts = Counter(votes)
        predicted = vote_counts.most_common(1)[0][0]

        actual_gen = YEAR_TO_GENERATION.get(actual, actual)
        pred_gen = YEAR_TO_GENERATION.get(predicted, predicted)

        forest_results.append({
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "gen_correct": actual_gen == pred_gen,
        })

    total = len(forest_results)
    res7 = {
        "year_acc": sum(r["correct"] for r in forest_results) / total,
        "gen_acc": sum(r["gen_correct"] for r in forest_results) / total,
        "year_correct": sum(r["correct"] for r in forest_results),
        "gen_correct": sum(r["gen_correct"] for r in forest_results),
        "total": total,
    }
    print(f"  Year accuracy:       {res7['year_acc']:.1%} ({res7['year_correct']}/{res7['total']})")
    print(f"  Generation accuracy: {res7['gen_acc']:.1%} ({res7['gen_correct']}/{res7['total']})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Forest + cropping — vote + crop to region of interest")
    print("=" * 70)
    forest_cropped_results = []
    for entry in manifest:
        path = Path(entry["processed_path"])
        actual = entry["label"].split("_")[0]

        votes = []
        for tree in forest:
            node = tree
            while node["type"] != "leaf":
                answer = ans2(path, node["question"])
                node = node["yes"] if answer else node["no"]
            votes.append(node["label"])

        vote_counts = Counter(votes)
        predicted = vote_counts.most_common(1)[0][0]

        actual_gen = YEAR_TO_GENERATION.get(actual, actual)
        pred_gen = YEAR_TO_GENERATION.get(predicted, predicted)

        forest_cropped_results.append({
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "gen_correct": actual_gen == pred_gen,
        })

    total = len(forest_cropped_results)
    res8 = {
        "year_acc": sum(r["correct"] for r in forest_cropped_results) / total,
        "gen_acc": sum(r["gen_correct"] for r in forest_cropped_results) / total,
        "year_correct": sum(r["correct"] for r in forest_cropped_results),
        "gen_correct": sum(r["gen_correct"] for r in forest_cropped_results),
        "total": total,
    }
    print(f"  Year accuracy:       {res8['year_acc']:.1%} ({res8['year_correct']}/{res8['total']})")
    print(f"  Generation accuracy: {res8['gen_acc']:.1%} ({res8['gen_correct']}/{res8['total']})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<45} {'Year':>6} {'Gen':>6}")
    print("-" * 60)
    experiments = [
        ("1. Baseline (standard key + CLIP)", res1),
        ("2. Cropped to region of interest", res2),
        ("3. Detailed prompts", res3),
        ("4. Cropped + detailed prompts", res4),
        ("5. Complete linkage clustering", res5),
        ("6. Average linkage clustering", res6),
        ("7. Random forest of 5 keys (vote)", res7),
        ("8. Forest + cropping", res8),
    ]
    for name, res in experiments:
        print(f"{name:<45} {res['year_acc']:>5.1%} {res['gen_acc']:>5.1%}")
    print(f"\n{'Random chance':<45} {'3.7%':>6} {'16.7%':>6}")
