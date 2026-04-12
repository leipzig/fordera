"""Feature extraction and classification for Ford F-series trucks."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import LeaveOneOut
import pickle


# Generation groupings for Ford F-1 / F-100
GENERATIONS = {
    "Gen 1 (1948-1952)": ["1948-1950", "1951", "1952"],
    "Gen 2 (1953-1956)": ["1953", "1954", "1955", "1956"],
    "Gen 3 (1957-1960)": ["1957", "1958", "1959", "1960"],
    "Gen 4 (1961-1966)": ["1961", "1962", "1963", "1964", "1965", "1966"],
    "Gen 5 (1967-1972)": ["1967", "1968", "1969", "1970", "1971", "1972"],
    "Gen 6 (1973-1979)": ["1973-1975", "1976-1977", "1978", "1979"],
}

# Build reverse mapping: year_label -> generation
YEAR_TO_GENERATION = {}
for gen, years in GENERATIONS.items():
    for y in years:
        YEAR_TO_GENERATION[y] = gen


# Image transforms for ResNet input
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augmentation transforms for training
AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor:
    """Extract embeddings from images using a pre-trained ResNet backbone."""

    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

    def extract(self, image_path: Path) -> np.ndarray:
        """Extract a feature vector from a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().numpy()

    def extract_augmented(self, image_path: Path, n_augments: int = 10) -> List[np.ndarray]:
        """Extract features from augmented versions of an image."""
        img = Image.open(image_path).convert("RGB")
        features = []
        for _ in range(n_augments):
            tensor = AUGMENT_TRANSFORM(img).unsqueeze(0)
            with torch.no_grad():
                feat = self.model(tensor)
            features.append(feat.squeeze().numpy())
        return features

    def get_backbone(self) -> nn.Module:
        """Return the full ResNet50 (with final layer) for Grad-CAM."""
        model = models.resnet50(pretrained=True)
        model.eval()
        return model


class TruckClassifier:
    """Classify Ford F-series trucks by model year using cosine-similarity k-NN."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.knn = KNeighborsClassifier(
            n_neighbors=1, metric="cosine", weights="distance"
        )
        self.label_encoder = LabelEncoder()
        self._is_trained = False
        self._train_features = None
        self._train_labels = None

    def train(self, manifest: List[dict], n_augments: int = 20) -> Dict:
        """Train the classifier on processed images with augmentation."""
        all_features = []
        all_labels = []

        for entry in manifest:
            path = Path(entry["processed_path"])
            label = entry["label"].split("_")[0]  # Remove _alt suffix

            # Original features
            feat = self.extractor.extract(path)
            all_features.append(feat)
            all_labels.append(label)

            # Augmented features
            aug_feats = self.extractor.extract_augmented(path, n_augments)
            all_features.extend(aug_feats)
            all_labels.extend([label] * n_augments)

        X = np.array(all_features)
        X = normalize(X)  # L2 normalize for cosine similarity
        self.label_encoder.fit(all_labels)
        y = self.label_encoder.transform(all_labels)

        self.knn.fit(X, y)
        self._train_features = X
        self._train_labels = y
        self._is_trained = True

        return {
            "n_samples": len(X),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
        }

    def predict(self, image_path: Path) -> Tuple[str, float, Dict[str, float]]:
        """Predict the model year of a truck image."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained yet")

        features = self.extractor.extract(image_path)
        features = normalize(features.reshape(1, -1))

        # Get distances and indices to all neighbors for probability estimate
        distances, indices = self.knn.kneighbors(features, n_neighbors=min(5, len(self._train_labels)))

        # Convert distances to similarities (cosine distance -> similarity)
        similarities = 1 - distances[0]

        # Weighted vote across neighbors
        class_scores = {}
        for sim, idx in zip(similarities, indices[0]):
            label = self.label_encoder.inverse_transform([self._train_labels[idx]])[0]
            class_scores[label] = class_scores.get(label, 0) + sim

        # Normalize to probabilities
        total = sum(class_scores.values())
        all_probs = {}
        for label in self.label_encoder.classes_:
            all_probs[label] = class_scores.get(label, 0) / total if total > 0 else 0

        pred_label = max(all_probs, key=all_probs.get)
        confidence = all_probs[pred_label]

        return pred_label, confidence, all_probs

    def evaluate_loo(self, manifest: List[dict]) -> Dict:
        """Leave-one-out cross-validation using cosine similarity on original embeddings.

        For each held-out image, find its nearest neighbor among the remaining
        original (non-augmented) images using cosine similarity.
        """
        entries = []
        for entry in manifest:
            label = entry["label"].split("_")[0]
            entries.append({"path": Path(entry["processed_path"]), "label": label})

        # Extract all original features
        features = []
        labels = []
        for e in entries:
            feat = self.extractor.extract(e["path"])
            features.append(feat)
            labels.append(e["label"])

        X = np.array(features)
        X = normalize(X)  # L2 normalize

        correct = 0
        total = 0
        results = []

        for i in range(len(X)):
            # Compute cosine similarity to all other images
            test_vec = X[i:i+1]
            sims = (X @ test_vec.T).squeeze()
            sims[i] = -1  # Exclude self

            # Find nearest neighbor
            nn_idx = np.argmax(sims)
            pred_label = labels[nn_idx]
            actual_label = labels[i]

            is_correct = pred_label == actual_label
            correct += is_correct
            total += 1

            results.append({
                "actual": actual_label,
                "predicted": pred_label,
                "correct": bool(is_correct),
                "similarity": float(sims[nn_idx]),
            })

        accuracy = correct / total
        unique_labels = list(set(labels))
        n_classes = len(unique_labels)
        random_chance = 1.0 / n_classes

        # Generation-level accuracy
        gen_correct = sum(
            1 for r in results
            if YEAR_TO_GENERATION.get(r["actual"]) == YEAR_TO_GENERATION.get(r["predicted"])
        )
        gen_accuracy = gen_correct / total
        n_generations = len(GENERATIONS)
        gen_random_chance = 1.0 / n_generations

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "n_classes": n_classes,
            "random_chance": random_chance,
            "above_chance": accuracy > random_chance,
            "generation_accuracy": gen_accuracy,
            "generation_correct": gen_correct,
            "n_generations": n_generations,
            "generation_random_chance": gen_random_chance,
            "generation_above_chance": gen_accuracy > gen_random_chance,
            "results": results,
        }

    def save(self, path: Path) -> None:
        """Save the trained classifier to disk."""
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "knn": self.knn,
            "label_encoder": self.label_encoder,
            "train_features": self._train_features,
            "train_labels": self._train_labels,
        }
        pickle.dump(state, open(path / "classifier.pkl", "wb"))

    def load(self, path: Path) -> None:
        """Load a trained classifier from disk."""
        state = pickle.load(open(path / "classifier.pkl", "rb"))
        self.knn = state["knn"]
        self.label_encoder = state["label_encoder"]
        self._train_features = state["train_features"]
        self._train_labels = state["train_labels"]
        self._is_trained = True


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    model_dir = Path(__file__).parent.parent.parent / "models"

    manifest = json.loads((proc_dir / "manifest.json").read_text())

    clf = TruckClassifier()

    # First, run LOO evaluation
    print("Running leave-one-out evaluation...")
    loo_results = clf.evaluate_loo(manifest)
    print(f"LOO accuracy: {loo_results['accuracy']:.1%} "
          f"({loo_results['correct']}/{loo_results['total']})")
    print(f"Random chance: {loo_results['random_chance']:.1%}")
    for r in loo_results["results"]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['actual']} -> {r['predicted']} (sim: {r['similarity']:.3f})")

    # Train full classifier
    print("\nTraining classifier...")
    stats = clf.train(manifest, n_augments=20)
    print(f"Trained on {stats['n_samples']} samples across {stats['n_classes']} classes")

    clf.save(model_dir)
    print(f"Model saved to {model_dir}")

    # Test prediction
    first = manifest[0]
    label, conf, probs = clf.predict(Path(first["processed_path"]))
    print(f"\nTest prediction: {first['label']} -> {label} (confidence: {conf:.2f})")
