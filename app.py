import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Ford F-Series Classifier")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Ford F-Series Pickup Classifier & Dichotomous Key

        Upload a front-profile photo of a **Ford F-1 (1948–1952)** or **F-100 (1953–1979)**
        pickup truck to identify its model year.

        The classifier uses a pre-trained ResNet50 backbone with cosine-similarity matching.
        Grad-CAM heatmaps show which visual features drove the classification.
        An automatically generated dichotomous key helps you identify trucks step-by-step.
        """
    )
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    MODEL_DIR = Path("models")
    DATA_DIR = Path("data/processed")
    OUTPUT_DIR = Path("outputs")

    return json, Path, cv2, np, Image, MODEL_DIR, DATA_DIR, OUTPUT_DIR


@app.cell
def _(MODEL_DIR, DATA_DIR, OUTPUT_DIR, json, Path):
    from fordera.classifier import TruckClassifier, GENERATIONS, YEAR_TO_GENERATION
    from fordera.interpretability import GradCAM, extract_zone_activations
    from fordera.keygen import DichotomousKeyGenerator

    # Load models
    classifier = TruckClassifier()
    classifier.load(MODEL_DIR)

    gradcam = GradCAM()

    keygen = DichotomousKeyGenerator()
    keygen.load(MODEL_DIR)

    # Load manifest
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())

    # Load interactive key
    key_json = keygen.to_interactive_json(manifest)

    return (
        classifier,
        gradcam,
        keygen,
        manifest,
        key_json,
        TruckClassifier,
        GENERATIONS,
        YEAR_TO_GENERATION,
        GradCAM,
        extract_zone_activations,
        DichotomousKeyGenerator,
    )


@app.cell
def _(mo):
    mo.md("## Upload a Truck Photo")
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg", ".webp"], label="Upload front-profile photo")
    upload
    return (upload,)


@app.cell
def _(upload, mo, np, cv2, Image, Path, classifier, gradcam, extract_zone_activations, YEAR_TO_GENERATION):
    import tempfile
    import io
    import base64

    def _img_to_data_uri(img_array, fmt="png"):
        """Convert a numpy BGR image to a data URI for display."""
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format=fmt.upper())
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt};base64,{b64}"

    if upload.value:
        file_data = upload.value[0]
        # Save to temp file for processing
        suffix = "." + file_data.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_data.contents)
            tmp_path = Path(f.name)

        # Classify
        pred_label, confidence, all_probs = classifier.predict(tmp_path)
        generation = YEAR_TO_GENERATION.get(pred_label, "Unknown")

        # Grad-CAM
        heatmap = gradcam.generate(tmp_path)
        overlay = gradcam.overlay(tmp_path, heatmap)
        zones = extract_zone_activations(heatmap)

        # Original image resized
        orig = cv2.imread(str(tmp_path))
        orig = cv2.resize(orig, (224, 224))

        orig_uri = _img_to_data_uri(orig)
        overlay_uri = _img_to_data_uri(overlay)

        # Top 5 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])[:5]
        prob_table = "\n".join(
            f"| {label} | {prob:.1%} |" for label, prob in sorted_probs
        )

        # Zone activations
        sorted_zones = sorted(zones.items(), key=lambda x: -x[1])
        zone_table = "\n".join(
            f"| {name} | {'█' * int(val * 20)} | {val:.2f} |"
            for name, val in sorted_zones
        )

        result = mo.md(f"""
## Classification Result

**Predicted Year:** {pred_label} ({generation})
**Confidence:** {confidence:.1%}

| Original | Grad-CAM Overlay |
|----------|-----------------|
| ![original]({orig_uri}) | ![gradcam]({overlay_uri}) |

### Top 5 Predictions

| Year | Probability |
|------|-------------|
{prob_table}

### Feature Zone Activations

| Zone | Activation | Score |
|------|------------|-------|
{zone_table}
""")
    else:
        result = mo.md("*Upload an image above to see the classification result.*")

    result
    return


@app.cell
def _(mo):
    mo.md("## Interactive Dichotomous Key")
    return


@app.cell
def _(mo, key_json, cv2, Image, Path, io, base64):
    import marimo as _mo

    def _build_key_ui(node, depth=0):
        """Recursively build the interactive key display."""
        if node["type"] == "leaf":
            label = node["label"]
            imgs_html = ""
            for img_path in node.get("example_images", [])[:2]:
                if Path(img_path).exists():
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img_rgb)
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        imgs_html += f'<img src="data:image/png;base64,{b64}" width="112" style="margin:2px;border:1px solid #ccc;border-radius:4px;">'

            return mo.md(f"""
### 🏁 **{label}**

{imgs_html}
""")

        question = node["question"]
        yes_content = _build_key_ui(node["yes"], depth + 1)
        no_content = _build_key_ui(node["no"], depth + 1)

        return mo.accordion({
            f"{'  ' * depth}❓ {question}": mo.vstack([
                mo.hstack([
                    mo.accordion({"✅ Yes": yes_content}),
                    mo.accordion({"❌ No": no_content}),
                ]),
            ]),
        })

    key_ui = _build_key_ui(key_json)
    key_ui
    return


@app.cell
def _(mo):
    mo.md("## Printable Key")
    return


@app.cell
def _(mo, Path):
    svg_path = Path("outputs/dichotomous_key.svg")
    pdf_path = Path("outputs/dichotomous_key.pdf")

    if svg_path.exists():
        svg_content = svg_path.read_text()
        download_links = ""
        if pdf_path.exists():
            download_links = mo.download(
                data=pdf_path.read_bytes(),
                filename="ford_f_series_key.pdf",
                mimetype="application/pdf",
                label="Download PDF",
            )

        printable = mo.vstack([
            download_links,
            mo.Html(svg_content),
        ])
    else:
        printable = mo.md("*Printable key not generated yet. Run the key generator first.*")

    printable
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        *Built with PyTorch, Grad-CAM, scikit-learn, and Marimo.
        Data from [Street Trucks Magazine](https://www.streettrucksmag.com/complete-history-of-the-ford-f-series-pickup/).*
        """
    )
    return


if __name__ == "__main__":
    app.run()
