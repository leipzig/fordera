# Plan: Ford F-Series Front Profile Classifier & Dichotomous Key

> Source PRD: [leipzig/fordera#1](https://github.com/leipzig/fordera/issues/1)

## Architectural decisions

Durable decisions that apply across all phases:

- **Stack**: PyTorch for model code, scikit-learn for decision tree, marimo for notebooks/app
- **Data directory**: `data/raw/` for scraped images, `data/processed/` for masked/resized images. Each image named `{year}.jpg` (or `{year}_{n}.jpg` if multiple per year)
- **Model artifacts**: `models/` directory for saved weights, embeddings, and decision tree
- **Key outputs**: `outputs/` directory for generated key (PDF/SVG, JSON)
- **Pre-trained backbone**: Frozen CNN (ResNet or EfficientNet) or CLIP image encoder — no fine-tuning of backbone weights
- **Classifier head**: Lightweight (SVM, k-NN, or small FC) on top of frozen embeddings
- **Interpretability**: Grad-CAM on the backbone's final convolutional layer
- **Key generation**: scikit-learn DecisionTreeClassifier on Grad-CAM-derived spatial features, rendered via graphviz
- **Text masking**: OCR detection (EasyOCR or pytesseract) → inpainting to remove year text overlays
- **Testing**: pytest, behavioral tests only (no implementation detail assertions)

---

## Phase 1: Data Acquisition & Exploration

**User stories**: 13, 7

### What to build

A scraper that downloads all front-profile images from the Street Trucks Mag page for the 1948–1979 F-1 and F-100 trucks. The scraper parses the page HTML, finds image URLs near year-related headings/text, and saves each image labeled with its model year. After scraping, an exploration notebook determines the actual class count — which years have distinct images, which share images, and which years may need to collapse into the same class. This exploration step produces a manifest (CSV or JSON) mapping each image file to its year label, which all downstream phases consume.

### Acceptance criteria

- [ ] Scraper downloads all 1948–1979 F-1/F-100 front-profile images from the page
- [ ] Each image is saved with its model year label in the filename
- [ ] A manifest file maps image paths to year labels
- [ ] Exploration notebook documents the actual number of distinct classes and any year-collapsing decisions
- [ ] The pipeline is reproducible in a marimo notebook

---

## Phase 2: Text Masking Pipeline + Tests

**User stories**: 8

### What to build

A preprocessing module that takes raw scraped images and removes year text overlays to prevent data leakage. The module uses OCR to detect text regions containing year numbers, then inpaints or fills those regions so the model cannot learn to read text. It also resizes and normalizes images for model input. Tests verify that text is actually removed, output dimensions are correct, and non-text regions are not corrupted.

### Acceptance criteria

- [ ] OCR detects year text regions in raw images
- [ ] Detected text regions are inpainted/masked — re-running OCR on output finds no year strings
- [ ] Output images are resized to a consistent dimension suitable for the chosen backbone
- [ ] Non-text regions of the image are preserved (pixel similarity check outside masked area)
- [ ] Preprocessor tests pass in pytest

---

## Phase 3: Feature Extraction & Classification + Tests

**User stories**: 1, 9, 10, 15

### What to build

A classification pipeline that takes preprocessed images, extracts embeddings via a frozen pre-trained CNN backbone, and trains a lightweight classifier head to predict model year. Data augmentation (rotation, flip, color jitter, random crop) is applied during training to stretch the small dataset. The classifier returns a predicted year and a confidence score. Evaluation uses leave-one-out cross-validation given the tiny dataset. Tests verify predictions are valid years, confidence scores are present, and accuracy exceeds random chance.

### Acceptance criteria

- [ ] Pre-trained backbone produces embedding vectors for each image
- [ ] Classifier predicts a valid model year (within the manifest's label set) for any input image
- [ ] Each prediction includes a confidence score
- [ ] Leave-one-out accuracy exceeds random chance for the number of classes
- [ ] Data augmentation is applied during training
- [ ] The model handles varying input image sizes gracefully
- [ ] Classifier tests pass in pytest

---

## Phase 4: Grad-CAM Interpretability

**User stories**: 2, 3

### What to build

An interpretability module that applies Grad-CAM to the backbone's final convolutional layer to produce spatial activation heatmaps for each prediction. The heatmap shows which image regions most influenced the classification. The module also clusters high-activation regions across images to identify recurring discriminative areas (e.g., grille zone, headlight zone, bumper zone) and assigns descriptive labels to these clusters. These labeled feature regions become the vocabulary for the dichotomous key in Phase 5.

### Acceptance criteria

- [ ] Grad-CAM produces a heatmap overlay for any input image + prediction
- [ ] Heatmaps highlight plausible truck front-end regions (not background noise)
- [ ] High-activation regions are clustered across images into named feature zones
- [ ] Each feature zone has a human-readable label (e.g., "grille area", "headlight shape")
- [ ] Feature zones and their activation patterns per class are saved for key generation

---

## Phase 5: Dichotomous Key Generation + Tests

**User stories**: 4, 5, 6, 11, 12, 14, 16

### What to build

A key generator that takes the Grad-CAM-derived feature zones and their activation patterns per class, encodes them as feature vectors, and trains a scikit-learn DecisionTreeClassifier. The decision tree is converted into two output formats: (1) a printable tree diagram (PDF/SVG via graphviz) with plain-language feature descriptions at each branch and example images at leaf nodes, and (2) a JSON structure for the interactive marimo widget. Tests verify the tree is well-formed, all years are reachable, and outputs are valid.

### Acceptance criteria

- [ ] Decision tree is trained on Grad-CAM-derived features
- [ ] Branch labels use plain-language visual descriptions, not numeric feature indices
- [ ] Every model year in the training set is reachable via some path through the tree
- [ ] Printable output is a valid PDF or SVG with a clean tree layout
- [ ] Interactive key JSON structure includes: node questions, yes/no branches, leaf year labels, and example image references
- [ ] Example images are shown at branching points / leaf nodes
- [ ] Key generator tests pass in pytest

---

## Phase 6: Marimo App Integration

**User stories**: 1, 2, 3, 4, 5, 7, 10, 14

### What to build

A single reactive marimo notebook that ties all modules together into a cohesive interactive app. The notebook includes: an image upload widget, the preprocessing → classification pipeline with year prediction and confidence display, a Grad-CAM overlay visualization panel, an interactive dichotomous key walkthrough (click through yes/no questions to reach a year), and a button to export/view the printable key. The notebook is the final deliverable — it serves as both the development environment and the end-user app.

### Acceptance criteria

- [ ] Image upload widget accepts a front-profile photo
- [ ] Uploaded image is preprocessed (text masked) and classified with year + confidence displayed
- [ ] Grad-CAM heatmap overlay is shown on the uploaded image
- [ ] Interactive dichotomous key is navigable via yes/no selections
- [ ] Example images appear at key branching points
- [ ] Printable key is viewable/downloadable as PDF or SVG
- [ ] The notebook runs end-to-end without errors on a fresh `marimo run`
