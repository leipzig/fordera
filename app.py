import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Ford F-Series Classifier")


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

        ## Upload a Truck Photo
        """
    )
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
def _(mo, key_json, cv2, Image, Path, io, base64):
    # Build the key HTML iteratively using a stack to avoid marimo
    # mangling recursive function names (underscore-prefixed functions
    # get renamed with a cell prefix, breaking self-calls).
    img_cache = {}

    def make_thumb(img_path, width):
        key = (img_path, width)
        if key in img_cache:
            return img_cache[key]
        p = Path(img_path)
        if not p.exists():
            img_cache[key] = ""
            return ""
        raw = cv2.imread(str(p))
        if raw is None:
            img_cache[key] = ""
            return ""
        thumb_size = min(width * 2, 120)
        raw = cv2.resize(raw, (thumb_size, thumb_size))
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode()
        tag = f'<img src="data:image/jpeg;base64,{b64}" width="{width}" style="margin:2px;border:1px solid #ccc;border-radius:4px;vertical-align:middle;">'
        img_cache[key] = tag
        return tag

    # Post-order iterative tree traversal: first collect all nodes,
    # then build HTML bottom-up so each parent can embed its children.
    nodes_in_order = []
    stack = [(key_json, 0)]
    while stack:
        node, depth = stack.pop()
        nodes_in_order.append((node, depth))
        if node["type"] == "decision":
            stack.append((node["no"], depth + 1))
            stack.append((node["yes"], depth + 1))

    # Reverse so leaves come first (post-order)
    nodes_in_order.reverse()
    html_map = {}

    for node, depth in nodes_in_order:
        node_id = id(node)
        if node["type"] == "leaf":
            label = node["label"]
            imgs = "".join(make_thumb(p, 80) for p in node.get("example_images", [])[:1])
            html_map[node_id] = f'<div style="margin:6px 0;padding:6px 10px;background:#e8f5e9;border-radius:6px;display:inline-block;"><strong>{label}</strong> {imgs}</div>'
        else:
            question = node["question"]
            yes_imgs = "".join(make_thumb(p, 56) for p in node.get("yes_images", [])[:1])
            no_imgs = "".join(make_thumb(p, 56) for p in node.get("no_images", [])[:1])
            yes_html = html_map[id(node["yes"])]
            no_html = html_map[id(node["no"])]
            open_attr = " open" if depth < 1 else ""
            html_map[node_id] = f"""<details{open_attr} style="margin:4px 0;border-left:2px solid #1976d2;padding-left:10px;">
  <summary style="cursor:pointer;font-weight:bold;padding:3px 0;font-size:14px;">{question}</summary>
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">
    <div style="flex:1;min-width:180px;">
      <div style="color:#2e7d32;font-weight:bold;margin-bottom:3px;font-size:13px;">Yes {yes_imgs}</div>
      {yes_html}
    </div>
    <div style="flex:1;min-width:180px;">
      <div style="color:#c62828;font-weight:bold;margin-bottom:3px;font-size:13px;">No {no_imgs}</div>
      {no_html}
    </div>
  </div>
</details>"""

    root_html = html_map[id(key_json)]
    full_html = f"<h2>Interactive Dichotomous Key</h2>\n<p><em>Click questions to expand. Each branch shows example truck photos.</em></p>\n{root_html}"
    mo.Html(full_html)
    return


@app.cell
def _(mo, Path):
    svg_path = Path("outputs/dichotomous_key.svg")
    pdf_path = Path("outputs/dichotomous_key.pdf")

    if svg_path.exists():
        svg_content = svg_path.read_text()
        # Wrap SVG in a scrollable container that fits the page
        wrapped_svg = f'<div style="overflow:auto;max-width:100%;border:1px solid #ddd;border-radius:8px;padding:10px;background:#fafafa;">{svg_content}</div>'

        parts = [mo.md("## Printable Key")]
        if pdf_path.exists():
            parts.append(
                mo.download(
                    data=pdf_path.read_bytes(),
                    filename="ford_f_series_key.pdf",
                    mimetype="application/pdf",
                    label="Download PDF",
                )
            )
        parts.append(mo.Html(wrapped_svg))
        printable = mo.vstack(parts)
    else:
        printable = mo.md("## Printable Key\n\n*Not generated yet. Run the key generator first.*")

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
