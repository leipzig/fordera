import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Ford F-Series Classifier")


@app.cell
def _():
    import marimo as mo
    import json
    import io
    import base64
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    MODEL_DIR = Path("models")
    DATA_DIR = Path("data/processed")
    OUTPUT_DIR = Path("outputs")

    return mo, json, io, base64, tempfile, Path, cv2, np, Image, MODEL_DIR, DATA_DIR, OUTPUT_DIR


@app.cell
def _(MODEL_DIR, DATA_DIR, json, Path):
    from fordera.classifier import TruckClassifier, GENERATIONS, YEAR_TO_GENERATION
    from fordera.interpretability import GradCAM, extract_zone_activations
    from fordera.keygen import DichotomousKeyGenerator

    classifier = TruckClassifier()
    classifier.load(MODEL_DIR)

    gradcam = GradCAM()

    keygen = DichotomousKeyGenerator()
    keygen.load(MODEL_DIR)

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    key_json = keygen.to_interactive_json(manifest)

    return (
        classifier, gradcam, keygen, manifest, key_json,
        YEAR_TO_GENERATION, extract_zone_activations,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Ford F-Series Pickup Classifier & Dichotomous Key

        Upload a front-profile photo of a **Ford F-1 (1948-1952)** or **F-100 (1953-1979)**
        pickup truck to identify its model year.

        The classifier uses a pre-trained ResNet50 backbone with cosine-similarity matching.
        Grad-CAM heatmaps show which visual features drove the classification.
        An automatically generated dichotomous key helps you identify trucks step-by-step.
        """
    )
    return


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
def _(upload, mo, cv2, Image, Path, io, base64, tempfile, classifier, gradcam, extract_zone_activations, YEAR_TO_GENERATION):
    def _img_to_data_uri(img_array, fmt="png"):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format=fmt.upper())
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt};base64,{b64}"

    if upload.value:
        file_data = upload.value[0]
        suffix = "." + file_data.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_data.contents)
            tmp_path = Path(f.name)

        pred_label, confidence, all_probs = classifier.predict(tmp_path)
        generation = YEAR_TO_GENERATION.get(pred_label, "Unknown")

        heatmap = gradcam.generate(tmp_path)
        overlay_img = gradcam.overlay(tmp_path, heatmap)
        zones = extract_zone_activations(heatmap)

        orig = cv2.imread(str(tmp_path))
        orig = cv2.resize(orig, (224, 224))

        orig_uri = _img_to_data_uri(orig)
        overlay_uri = _img_to_data_uri(overlay_img)

        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])[:5]
        prob_table = "\n".join(
            f"| {label} | {prob:.1%} |" for label, prob in sorted_probs
        )

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
    def _img_tag(img_path, width=100):
        """Convert an image file to an inline HTML img tag."""
        if not Path(img_path).exists():
            return ""
        img = cv2.imread(str(img_path))
        if img is None:
            return ""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{width}" style="margin:2px;border:1px solid #ccc;border-radius:4px;">'

    def _build_key_ui(node, depth=0):
        if node["type"] == "leaf":
            label = node["label"]
            imgs_html = "".join(
                _img_tag(p, 112) for p in node.get("example_images", [])[:2]
            )
            return mo.md(f"**{label}**\n\n{imgs_html}")

        question = node["question"]

        # Build example images for yes/no sides
        yes_imgs = "".join(
            _img_tag(p, 80) for p in node.get("yes_images", [])[:3]
        )
        no_imgs = "".join(
            _img_tag(p, 80) for p in node.get("no_images", [])[:3]
        )

        yes_content = _build_key_ui(node["yes"], depth + 1)
        no_content = _build_key_ui(node["no"], depth + 1)

        return mo.accordion({
            f"{question}": mo.vstack([
                mo.hstack([
                    mo.vstack([
                        mo.md(f'**Yes** {yes_imgs}'),
                        mo.accordion({"Explore Yes branch": yes_content}),
                    ]),
                    mo.vstack([
                        mo.md(f'**No** {no_imgs}'),
                        mo.accordion({"Explore No branch": no_content}),
                    ]),
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
        parts = []
        if pdf_path.exists():
            parts.append(
                mo.download(
                    data=pdf_path.read_bytes(),
                    filename="ford_f_series_key.pdf",
                    mimetype="application/pdf",
                    label="Download PDF",
                )
            )
        parts.append(mo.Html(svg_content))
        printable = mo.vstack(parts)
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
