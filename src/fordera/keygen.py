"""Dichotomous key generator using hierarchical clustering + CLIP descriptions."""

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.preprocessing import LabelEncoder, normalize
import graphviz

from fordera.classifier import FeatureExtractor


class DichotomousKeyGenerator:
    """Generate a balanced dichotomous key via hierarchical clustering of embeddings."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tree_root = None  # dict tree structure
        self.node_descriptions = {}  # node_id -> English question
        self.node_examples = {}  # node_id -> {"yes_images": [...], "no_images": [...]}
        self._all_labels = []
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray, labels: List[str]) -> None:
        """Build a balanced binary tree via agglomerative clustering."""
        # Deduplicate: average embeddings per unique base label
        base_labels = [l.split("_")[0] for l in labels]
        unique_labels = sorted(set(base_labels))
        avg_embeddings = []
        for ul in unique_labels:
            mask = [i for i, bl in enumerate(base_labels) if bl == ul]
            avg_embeddings.append(embeddings[mask].mean(axis=0))
        avg_embeddings = np.array(avg_embeddings)
        avg_embeddings = normalize(avg_embeddings)

        self._all_labels = unique_labels

        # Hierarchical clustering using Ward's method
        Z = linkage(avg_embeddings, method="ward")
        root_node = to_tree(Z)

        # Convert scipy tree to our dict tree
        node_counter = [0]

        def _convert(node) -> Dict[str, Any]:
            nid = node_counter[0]
            node_counter[0] += 1

            if node.is_leaf():
                return {
                    "type": "leaf",
                    "node_id": nid,
                    "label": unique_labels[node.id],
                }
            else:
                return {
                    "type": "decision",
                    "node_id": nid,
                    "left": _convert(node.get_left()),
                    "right": _convert(node.get_right()),
                }

        self.tree_root = _convert(root_node)
        self._is_fitted = True

    def _collect_leaves(self, node: Dict) -> List[str]:
        """Collect all labels reachable from a node."""
        if node["type"] == "leaf":
            return [node["label"]]
        return self._collect_leaves(node["left"]) + self._collect_leaves(node["right"])

    def generate_descriptions(self, manifest: List[dict]) -> None:
        """Use CLIP to generate English descriptions for each split.

        Each node avoids reusing questions from its ancestors so the key
        has diverse, meaningful questions at every level.
        """
        from fordera.describer import CLIPDescriber

        label_to_paths = {}
        for entry in manifest:
            base_label = entry["label"].split("_")[0]
            path = Path(entry["processed_path"])
            if base_label not in label_to_paths:
                label_to_paths[base_label] = []
            label_to_paths[base_label].append(path)

        describer = CLIPDescriber()
        print("Generating CLIP descriptions for each split...")

        def _describe(node: Dict, ancestor_questions: set = None):
            if ancestor_questions is None:
                ancestor_questions = set()
            if node["type"] == "leaf":
                return

            left_labels = self._collect_leaves(node["left"])
            right_labels = self._collect_leaves(node["right"])

            left_paths = []
            for label in left_labels:
                left_paths.extend(label_to_paths.get(label, []))
            right_paths = []
            for label in right_labels:
                right_paths.extend(label_to_paths.get(label, []))

            question, desc, score = describer.best_distinguishing_feature(
                left_paths[:10], right_paths[:10],
                excluded_questions=ancestor_questions,
            )
            self.node_descriptions[node["node_id"]] = question
            self.node_examples[node["node_id"]] = {
                "yes_images": [str(p) for p in left_paths[:3]],
                "no_images": [str(p) for p in right_paths[:3]],
            }

            left_summary = ", ".join(left_labels[:4])
            right_summary = ", ".join(right_labels[:4])
            if len(left_labels) > 4:
                left_summary += f"... ({len(left_labels)} total)"
            if len(right_labels) > 4:
                right_summary += f"... ({len(right_labels)} total)"
            print(f"  Node {node['node_id']}: {question} (score: {score:.3f})")
            print(f"    Yes: [{left_summary}]")
            print(f"    No:  [{right_summary}]")

            child_excluded = ancestor_questions | {question}
            _describe(node["left"], child_excluded)
            _describe(node["right"], child_excluded)

        _describe(self.tree_root)

    def _question_for_node(self, node_id: int) -> str:
        return self.node_descriptions.get(node_id, "Visual difference?")

    def get_tree_text(self) -> str:
        """Return a text representation of the key."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")

        lines = []

        def _print(node, indent=""):
            if node["type"] == "leaf":
                lines.append(f"{indent}-> {node['label']}")
            else:
                question = self._question_for_node(node["node_id"])
                lines.append(f"{indent}{question}")
                lines.append(f"{indent}  Yes:")
                _print(node["left"], indent + "    ")
                lines.append(f"{indent}  No:")
                _print(node["right"], indent + "    ")

        _print(self.tree_root)
        return "\n".join(lines)

    def to_interactive_json(self, manifest: List[dict] = None) -> Dict[str, Any]:
        """Convert to JSON structure for interactive use."""
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

        def _build(node):
            if node["type"] == "leaf":
                return {
                    "type": "leaf",
                    "label": node["label"],
                    "example_images": image_lookup.get(node["label"], []),
                }
            question = self._question_for_node(node["node_id"])
            examples = self.node_examples.get(node["node_id"], {})
            return {
                "type": "decision",
                "question": question,
                "yes_images": examples.get("yes_images", []),
                "no_images": examples.get("no_images", []),
                "yes": _build(node["left"]),
                "no": _build(node["right"]),
            }

        return _build(self.tree_root)

    def to_graphviz(self, manifest: List[dict] = None) -> graphviz.Digraph:
        """Generate a graphviz tree diagram."""
        if not self._is_fitted:
            raise RuntimeError("Key not generated yet")

        dot = graphviz.Digraph(
            comment="Ford F-Series Dichotomous Key",
            format="svg",
        )
        dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
        dot.attr("edge", fontname="Helvetica", fontsize="10")

        def _add(node):
            nid = str(node["node_id"])
            if node["type"] == "leaf":
                dot.node(nid, node["label"], fillcolor="#90EE90",
                         shape="ellipse", style="filled", fontsize="11")
            else:
                question = self._question_for_node(node["node_id"])
                dot.node(nid, question, fillcolor="#ADD8E6", fontsize="9")
                dot.edge(nid, str(node["left"]["node_id"]), label="Yes")
                dot.edge(nid, str(node["right"]["node_id"]), label="No")
                _add(node["left"])
                _add(node["right"])

        _add(self.tree_root)
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
        return sorted(self._collect_leaves(self.tree_root))

    def save(self, path: Path) -> None:
        import pickle
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "tree_root": self.tree_root,
            "all_labels": self._all_labels,
            "node_descriptions": self.node_descriptions,
            "node_examples": self.node_examples,
        }
        with open(path / "keygen.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        import pickle
        with open(path / "keygen.pkl", "rb") as f:
            state = pickle.load(f)
        self.tree_root = state["tree_root"]
        self._all_labels = state["all_labels"]
        self.node_descriptions = state.get("node_descriptions", {})
        self.node_examples = state.get("node_examples", {})
        self._is_fitted = True


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    model_dir = Path(__file__).parent.parent.parent / "models"

    manifest = json.loads((proc_dir / "manifest.json").read_text())

    # Extract ResNet embeddings
    print("Extracting ResNet embeddings...")
    extractor = FeatureExtractor()
    embeddings = []
    labels = []
    for entry in manifest:
        feat = extractor.extract(Path(entry["processed_path"]))
        embeddings.append(feat)
        labels.append(entry["label"])
    embeddings = np.array(embeddings)
    print(f"Embeddings: {embeddings.shape}")

    # Build tree
    keygen = DichotomousKeyGenerator()
    keygen.fit(embeddings, labels)

    reachable = keygen.get_all_labels()
    all_labels = sorted(set(l.split("_")[0] for l in labels))
    print(f"Reachable labels: {len(reachable)}/{len(all_labels)}")

    # Generate CLIP descriptions
    keygen.generate_descriptions(manifest)

    # Print
    print("\nDichotomous Key:")
    print(keygen.get_tree_text())

    # Save
    key_json = keygen.to_interactive_json(manifest)
    json_path = output_dir / "dichotomous_key.json"
    json_path.write_text(json.dumps(key_json, indent=2))
    print(f"\nInteractive key saved to {json_path}")

    svg_path = keygen.render_printable(output_dir / "dichotomous_key", manifest)
    print(f"Printable key saved to {svg_path}")

    keygen.save(model_dir)
    print(f"Key generator saved to {model_dir}")
