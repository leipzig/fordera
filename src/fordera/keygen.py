"""Dichotomous key generator from ResNet embeddings + Grad-CAM interpretation."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import graphviz

from fordera.interpretability import (
    FEATURE_ZONES,
    describe_feature_zones,
    GradCAM,
    extract_zone_activations,
)
from fordera.classifier import FeatureExtractor


def _zone_description(zone_name: str) -> str:
    """Convert zone name to human-readable description."""
    descriptions = describe_feature_zones()
    return descriptions.get(zone_name, zone_name)


def _interpret_split(
    pca: PCA,
    feature_idx: int,
    zone_names: List[str],
    gradcam_features: np.ndarray,
    pca_features: np.ndarray,
) -> str:
    """Interpret a PCA-based split as a human-readable visual feature.

    Find which Grad-CAM zone correlates most with the PCA component used in this split.
    """
    # Correlate this PCA component with each zone activation
    pca_col = pca_features[:, feature_idx]
    best_zone = None
    best_corr = 0
    for i, zone_name in enumerate(zone_names):
        zone_col = gradcam_features[:, i]
        if zone_col.std() > 0 and pca_col.std() > 0:
            corr = abs(np.corrcoef(pca_col, zone_col)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_zone = zone_name

    if best_zone:
        return _zone_description(best_zone)
    return "visual feature"


class DichotomousKeyGenerator:
    """Generate a dichotomous key from ResNet embeddings with Grad-CAM interpretation."""

    def __init__(self):
        self.tree = DecisionTreeClassifier(
            min_samples_leaf=1,
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=20, random_state=42)
        self.zone_names = None
        self.gradcam_features = None
        self.pca_features = None
        self._is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        gradcam_features: np.ndarray,
        zone_names: List[str],
    ) -> None:
        """Fit the decision tree on PCA-reduced ResNet embeddings.

        Args:
            embeddings: ResNet embeddings (n_images, 2048)
            labels: Year labels per image
            gradcam_features: Zone activation features (n_images, n_zones) for interpretation
            zone_names: Names of the Grad-CAM zones
        """
        self.zone_names = zone_names
        self.gradcam_features = gradcam_features

        # Deduplicate: average embeddings per unique base label
        base_labels = [l.split("_")[0] for l in labels]
        unique_labels = sorted(set(base_labels))
        avg_embeddings = []
        avg_gradcam = []
        for ul in unique_labels:
            mask = [i for i, bl in enumerate(base_labels) if bl == ul]
            avg_embeddings.append(embeddings[mask].mean(axis=0))
            avg_gradcam.append(gradcam_features[mask].mean(axis=0))

        avg_embeddings = np.array(avg_embeddings)
        avg_gradcam = np.array(avg_gradcam)
        self.gradcam_features = avg_gradcam

        # PCA reduce
        n_components = min(20, len(unique_labels) - 1, avg_embeddings.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_features = self.pca.fit_transform(avg_embeddings)

        self.label_encoder.fit(unique_labels)
        y = self.label_encoder.transform(unique_labels)
        self.tree.fit(self.pca_features, y)
        self._is_fitted = True

    def _question_for_node(self, node_id: int) -> str:
        """Generate a human-readable question for a decision node."""
        tree = self.tree.tree_
        feature_idx = tree.feature[node_id]
        zone_desc = _interpret_split(
            self.pca, feature_idx, self.zone_names,
            self.gradcam_features, self.pca_features,
        )
        return f"Distinctive {zone_desc}?"

    def get_tree_text(self) -> str:
        """Return a text representation of the decision tree."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")
        feature_names = [f"PC{i}" for i in range(self.pca_features.shape[1])]
        return export_text(
            self.tree,
            feature_names=feature_names,
            class_names=list(self.label_encoder.classes_),
        )

    def to_interactive_json(self, manifest: List[dict] = None) -> Dict[str, Any]:
        """Convert the decision tree to a JSON structure for interactive use."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")

        image_lookup = {}
        if manifest:
            for entry in manifest:
                base_label = entry["label"].split("_")[0]
                if base_label not in image_lookup:
                    image_lookup[base_label] = []
                image_lookup[base_label].append(
                    entry.get("processed_path", entry.get("path"))
                )

        tree = self.tree.tree_
        classes = list(self.label_encoder.classes_)

        def _build_node(node_id: int) -> Dict[str, Any]:
            if tree.children_left[node_id] == tree.children_right[node_id]:
                class_idx = tree.value[node_id].argmax()
                label = classes[class_idx]
                return {
                    "type": "leaf",
                    "label": label,
                    "example_images": image_lookup.get(label, []),
                }

            question = self._question_for_node(node_id)
            return {
                "type": "decision",
                "question": question,
                "yes": _build_node(tree.children_left[node_id]),
                "no": _build_node(tree.children_right[node_id]),
            }

        return _build_node(0)

    def to_graphviz(self, manifest: List[dict] = None) -> graphviz.Digraph:
        """Generate a graphviz tree diagram for the dichotomous key."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")

        tree = self.tree.tree_
        classes = list(self.label_encoder.classes_)

        dot = graphviz.Digraph(
            comment="Ford F-Series Dichotomous Key",
            format="svg",
        )
        dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
        dot.attr("edge", fontname="Helvetica", fontsize="10")

        def _add_node(node_id: int):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                class_idx = tree.value[node_id].argmax()
                label = classes[class_idx]
                dot.node(
                    str(node_id),
                    label,
                    fillcolor="#90EE90",
                    shape="ellipse",
                    style="filled",
                    fontsize="11",
                )
            else:
                question = self._question_for_node(node_id)
                dot.node(
                    str(node_id),
                    question,
                    fillcolor="#ADD8E6",
                    fontsize="10",
                )

                left = tree.children_left[node_id]
                right = tree.children_right[node_id]
                dot.edge(str(node_id), str(left), label="Yes")
                dot.edge(str(node_id), str(right), label="No")
                _add_node(left)
                _add_node(right)

        _add_node(0)
        return dot

    def render_printable(self, output_path: Path, manifest: List[dict] = None) -> Path:
        """Render the key as SVG and PDF."""
        dot = self.to_graphviz(manifest)
        svg_path = output_path.with_suffix("")
        dot.render(str(svg_path), format="svg", cleanup=True)
        dot.render(str(svg_path), format="pdf", cleanup=True)
        return Path(str(svg_path) + ".svg")

    def get_all_labels(self) -> List[str]:
        """Return all class labels reachable in the tree."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")

        tree = self.tree.tree_
        classes = list(self.label_encoder.classes_)
        labels = set()

        def _collect(node_id):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                class_idx = tree.value[node_id].argmax()
                labels.add(classes[class_idx])
            else:
                _collect(tree.children_left[node_id])
                _collect(tree.children_right[node_id])

        _collect(0)
        return sorted(labels)

    def save(self, path: Path) -> None:
        """Save the key generator state."""
        import pickle
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "tree": self.tree,
            "label_encoder": self.label_encoder,
            "pca": self.pca,
            "zone_names": self.zone_names,
            "gradcam_features": self.gradcam_features,
            "pca_features": self.pca_features,
        }
        with open(path / "keygen.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        """Load the key generator state."""
        import pickle
        with open(path / "keygen.pkl", "rb") as f:
            state = pickle.load(f)
        self.tree = state["tree"]
        self.label_encoder = state["label_encoder"]
        self.pca = state["pca"]
        self.zone_names = state["zone_names"]
        self.gradcam_features = state["gradcam_features"]
        self.pca_features = state["pca_features"]
        self._is_fitted = True


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    model_dir = Path(__file__).parent.parent.parent / "models"

    manifest = json.loads((proc_dir / "manifest.json").read_text())

    # Load Grad-CAM features
    data = np.load(output_dir / "gradcam_features.npz", allow_pickle=True)
    gradcam_features = data["features"]
    labels = list(data["labels"])
    zone_names = list(data["zone_names"])

    # Extract ResNet embeddings
    print("Extracting ResNet embeddings...")
    extractor = FeatureExtractor()
    embeddings = []
    for entry in manifest:
        feat = extractor.extract(Path(entry["processed_path"]))
        embeddings.append(feat)
    embeddings = np.array(embeddings)
    print(f"Embeddings: {embeddings.shape}")

    # Generate key
    keygen = DichotomousKeyGenerator()
    keygen.fit(embeddings, labels, gradcam_features, zone_names)

    # Print text tree
    print("\nDecision Tree:")
    print(keygen.get_tree_text())

    # Check coverage
    reachable = keygen.get_all_labels()
    all_labels = sorted(set(l.split("_")[0] for l in labels))
    print(f"\nReachable labels: {len(reachable)}/{len(all_labels)}")
    missing = set(all_labels) - set(reachable)
    if missing:
        print(f"Missing: {missing}")
    else:
        print("All labels reachable!")

    # Save interactive JSON
    key_json = keygen.to_interactive_json(manifest)
    json_path = output_dir / "dichotomous_key.json"
    json_path.write_text(json.dumps(key_json, indent=2))
    print(f"\nInteractive key saved to {json_path}")

    # Render printable
    svg_path = keygen.render_printable(output_dir / "dichotomous_key", manifest)
    print(f"Printable key saved to {svg_path}")

    keygen.save(model_dir)
    print(f"Key generator saved to {model_dir}")
