"""Use CLIP to generate English descriptions for decision tree splits.

For each split in the dichotomous key, compares images on each side
against a vocabulary of truck visual features to find the description
that best separates the two groups.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import clip
import numpy as np
import torch
from PIL import Image


# Vocabulary of visual features observable on Ford F-series front profiles.
# Each entry is (feature_description, opposite_description) so the question
# can be phrased as "Does it have [feature]?" with Yes/No answers.
FEATURE_VOCABULARY = [
    # Grille patterns
    ("a horizontal bar grille", "a mesh or egg-crate grille"),
    ("a single wide horizontal grille bar", "multiple thin grille bars"),
    ("a tall narrow grille opening", "a wide short grille opening"),
    ("a chrome grille surround", "a painted grille surround"),
    ("a two-tier split grille", "a single unified grille"),
    ("a honeycomb pattern grille", "a bar pattern grille"),
    ("a wide flat grille", "a narrow protruding grille"),
    ("a grille with vertical bars", "a grille with horizontal bars"),
    ("a recessed grille", "a flush or protruding grille"),
    # Headlights
    ("round headlights", "rectangular headlights"),
    ("single headlights on each side", "dual stacked headlights on each side"),
    ("quad headlights", "dual headlights"),
    ("headlights integrated into the grille", "headlights separate from the grille"),
    ("headlights mounted high on the fenders", "headlights mounted low near the bumper"),
    ("large prominent headlights", "small recessed headlights"),
    # Bumper
    ("a chrome front bumper", "a painted or body-color bumper"),
    ("a wide flat bumper", "a narrow wraparound bumper"),
    ("a bumper with integrated parking lights", "a bumper without parking lights"),
    ("a heavy duty bumper", "a light delicate bumper"),
    # Hood and body
    ("a flat hood", "a rounded or curved hood"),
    ("a long pointed hood", "a short blunt hood"),
    ("a hood with a center crease", "a smooth hood"),
    ("a forward-leaning front end", "an upright front end"),
    ("a boxy angular front end", "a rounded curvy front end"),
    ("a wide flat front end", "a narrow tall front end"),
    # Fenders
    ("separate fenders from the body", "fenders integrated into the body"),
    ("pronounced rounded fenders", "flat slab-sided fenders"),
    ("fender-mounted turn signals", "bumper-mounted turn signals"),
    # Era indicators
    ("a streamlined art-deco influenced design", "a utilitarian squared-off design"),
    ("a modern aerodynamic front end", "a classic flat front end"),
    ("visible Ford lettering on the hood", "no visible Ford lettering on the hood"),
    ("a wraparound windshield", "a flat windshield"),
    ("a two-piece split windshield", "a single-piece windshield"),
    ("parking lights above the headlights", "parking lights below the headlights"),
    ("a horizontal character line on the fenders", "no character line on the fenders"),
]


class CLIPDescriber:
    """Use CLIP to find the best English description for a set of images."""

    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_images(self, image_paths: List[Path]) -> torch.Tensor:
        """Encode a list of images into CLIP embeddings."""
        images = []
        for p in image_paths:
            img = self.preprocess(Image.open(p).convert("RGB"))
            images.append(img)
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text descriptions into CLIP embeddings."""
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def rank_distinguishing_features(
        self,
        left_paths: List[Path],
        right_paths: List[Path],
    ) -> List[Tuple[str, str, float]]:
        """Rank all feature descriptions by how well they separate two groups.

        Returns:
            List of (question, description, score) sorted by score descending
        """
        if not left_paths or not right_paths:
            return [("Visual difference?", "", 0.0)]

        left_features = self._encode_images(left_paths)
        right_features = self._encode_images(right_paths)

        left_mean = left_features.mean(dim=0, keepdim=True)
        right_mean = right_features.mean(dim=0, keepdim=True)

        scored = []

        for feat_yes, feat_no in FEATURE_VOCABULARY:
            prompt_yes = f"a pickup truck with {feat_yes}"
            prompt_no = f"a pickup truck with {feat_no}"
            text_features = self._encode_texts([prompt_yes, prompt_no])

            left_sim_yes = (left_mean @ text_features[0:1].T).item()
            left_sim_no = (left_mean @ text_features[1:2].T).item()
            right_sim_yes = (right_mean @ text_features[0:1].T).item()
            right_sim_no = (right_mean @ text_features[1:2].T).item()

            score_forward = (left_sim_yes - left_sim_no) - (right_sim_yes - right_sim_no)
            score_reverse = (right_sim_yes - right_sim_no) - (left_sim_yes - left_sim_no)

            if score_forward >= score_reverse:
                scored.append((f"Does it have {feat_yes}?", feat_yes, score_forward))
            else:
                scored.append((f"Does it have {feat_yes}?", feat_yes, score_reverse))

        scored.sort(key=lambda x: -x[2])
        return scored

    def best_distinguishing_feature(
        self,
        left_paths: List[Path],
        right_paths: List[Path],
        excluded_questions: set = None,
    ) -> Tuple[str, str, float]:
        """Find the best feature description that hasn't been used by ancestors.

        Args:
            left_paths: Image paths for the "yes" side
            right_paths: Image paths for the "no" side
            excluded_questions: Set of question strings already used by ancestor nodes

        Returns:
            (question, description, score)
        """
        if excluded_questions is None:
            excluded_questions = set()

        ranked = self.rank_distinguishing_features(left_paths, right_paths)
        for question, desc, score in ranked:
            if question not in excluded_questions:
                return question, desc, score

        # Fallback if all are excluded
        return ranked[0] if ranked else ("Visual difference?", "", 0.0)


def describe_all_splits(
    tree_obj,
    label_encoder,
    label_to_paths: Dict[str, List[Path]],
    describer: CLIPDescriber,
) -> Dict[int, str]:
    """Generate English descriptions for every decision node in the tree.

    Args:
        tree_obj: fitted DecisionTreeClassifier
        label_encoder: LabelEncoder mapping indices to year labels
        label_to_paths: dict mapping year label to list of image paths
        describer: CLIPDescriber instance

    Returns:
        dict mapping node_id to question string
    """
    tree = tree_obj.tree_
    classes = list(label_encoder.classes_)
    descriptions = {}

    def _collect_leaves(node_id) -> List[str]:
        """Collect all class labels reachable from a node."""
        if tree.children_left[node_id] == tree.children_right[node_id]:
            class_idx = tree.value[node_id].argmax()
            return [classes[class_idx]]
        left_labels = _collect_leaves(tree.children_left[node_id])
        right_labels = _collect_leaves(tree.children_right[node_id])
        return left_labels + right_labels

    def _describe_node(node_id):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            return  # leaf

        left_labels = _collect_leaves(tree.children_left[node_id])
        right_labels = _collect_leaves(tree.children_right[node_id])

        # Collect image paths for each side
        left_paths = []
        for label in left_labels:
            left_paths.extend(label_to_paths.get(label, []))
        right_paths = []
        for label in right_labels:
            right_paths.extend(label_to_paths.get(label, []))

        # Limit to avoid too many images (CLIP batching)
        left_paths = left_paths[:8]
        right_paths = right_paths[:8]

        question, desc, score = describer.best_distinguishing_feature(
            left_paths, right_paths
        )
        descriptions[node_id] = question
        print(f"  Node {node_id}: {question} (score: {score:.3f})")
        print(f"    Left ({len(left_labels)} classes): {left_labels[:5]}...")
        print(f"    Right ({len(right_labels)} classes): {right_labels[:5]}...")

        _describe_node(tree.children_left[node_id])
        _describe_node(tree.children_right[node_id])

    _describe_node(0)
    return descriptions
