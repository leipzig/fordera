"""Grad-CAM interpretability for discovering distinguishing visual features."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Spatial regions of a front-profile truck image (approximate zones)
# These map Grad-CAM activation clusters to human-readable feature names
FEATURE_ZONES = {
    "hood_top": {"y_range": (0.0, 0.2), "x_range": (0.1, 0.9), "description": "hood and roof line"},
    "upper_grille": {"y_range": (0.2, 0.4), "x_range": (0.2, 0.8), "description": "upper grille area"},
    "lower_grille": {"y_range": (0.4, 0.6), "x_range": (0.2, 0.8), "description": "lower grille area"},
    "headlights_left": {"y_range": (0.2, 0.5), "x_range": (0.0, 0.2), "description": "left headlight area"},
    "headlights_right": {"y_range": (0.2, 0.5), "x_range": (0.8, 1.0), "description": "right headlight area"},
    "turn_signals_left": {"y_range": (0.5, 0.65), "x_range": (0.0, 0.25), "description": "left turn signal area"},
    "turn_signals_right": {"y_range": (0.5, 0.65), "x_range": (0.75, 1.0), "description": "right turn signal area"},
    "bumper_center": {"y_range": (0.6, 0.75), "x_range": (0.2, 0.8), "description": "center bumper"},
    "bumper_left": {"y_range": (0.6, 0.75), "x_range": (0.0, 0.2), "description": "left bumper edge"},
    "bumper_right": {"y_range": (0.6, 0.75), "x_range": (0.8, 1.0), "description": "right bumper edge"},
    "fender_left": {"y_range": (0.25, 0.55), "x_range": (0.0, 0.15), "description": "left fender"},
    "fender_right": {"y_range": (0.25, 0.55), "x_range": (0.85, 1.0), "description": "right fender"},
    "lower_body": {"y_range": (0.75, 1.0), "x_range": (0.0, 1.0), "description": "lower body / undercarriage"},
}


class GradCAM:
    """Grad-CAM for ResNet50 — highlights regions that drive classification."""

    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self._activations = None
        self._gradients = None

        # Hook the last convolutional layer (layer4)
        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image_path: Path, target_class: int = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap for an image.

        Args:
            image_path: Path to input image
            target_class: ImageNet class index to explain (None = predicted class)

        Returns:
            Heatmap as numpy array (224x224), values 0-1
        """
        img = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)
        tensor.requires_grad_(True)

        # Forward pass
        output = self.model(tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (224, 224))
        return cam

    def overlay(self, image_path: Path, heatmap: np.ndarray) -> np.ndarray:
        """Overlay a Grad-CAM heatmap on the original image.

        Returns:
            BGR image with heatmap overlay
        """
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (224, 224))

        # Convert heatmap to color
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        return overlay


def extract_zone_activations(heatmap: np.ndarray) -> Dict[str, float]:
    """Extract mean activation per spatial zone from a Grad-CAM heatmap.

    Returns dict mapping zone name to mean activation (0-1).
    """
    h, w = heatmap.shape
    activations = {}

    for zone_name, zone in FEATURE_ZONES.items():
        y_start = int(zone["y_range"][0] * h)
        y_end = int(zone["y_range"][1] * h)
        x_start = int(zone["x_range"][0] * w)
        x_end = int(zone["x_range"][1] * w)

        region = heatmap[y_start:y_end, x_start:x_end]
        activations[zone_name] = float(region.mean())

    return activations


def build_feature_matrix(
    manifest: List[dict], gradcam: GradCAM
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a feature matrix of zone activations for all images.

    Returns:
        (feature_matrix, labels, zone_names)
        feature_matrix: shape (n_images, n_zones)
    """
    zone_names = list(FEATURE_ZONES.keys())
    features = []
    labels = []

    for entry in manifest:
        path = Path(entry["processed_path"])
        label = entry["label"].split("_")[0]

        heatmap = gradcam.generate(path)
        activations = extract_zone_activations(heatmap)

        feature_vec = [activations[z] for z in zone_names]
        features.append(feature_vec)
        labels.append(label)

    return np.array(features), labels, zone_names


def describe_feature_zones() -> Dict[str, str]:
    """Return human-readable descriptions for each feature zone."""
    return {name: zone["description"] for name, zone in FEATURE_ZONES.items()}


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((proc_dir / "manifest.json").read_text())

    gradcam = GradCAM()

    print("Generating Grad-CAM heatmaps and zone activations...")
    feature_matrix, labels, zone_names = build_feature_matrix(manifest, gradcam)

    print(f"\nFeature matrix: {feature_matrix.shape}")
    print(f"Zones: {zone_names}")

    # Save feature matrix for key generation
    np.savez(
        output_dir / "gradcam_features.npz",
        features=feature_matrix,
        labels=np.array(labels),
        zone_names=np.array(zone_names),
    )
    print(f"Features saved to {output_dir / 'gradcam_features.npz'}")

    # Generate and save overlay images for a few examples
    for entry in manifest[:5]:
        path = Path(entry["processed_path"])
        heatmap = gradcam.generate(path)
        overlay = gradcam.overlay(path, heatmap)

        out_path = output_dir / f"gradcam_{entry['label']}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"Saved overlay: {out_path}")

        activations = extract_zone_activations(heatmap)
        print(f"  {entry['label']} zones: ", end="")
        for zone, val in activations.items():
            print(f"{zone}={val:.2f} ", end="")
        print()
