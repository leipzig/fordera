"""Generate a feature vocabulary by asking a vision-language model to
describe each truck illustration, then extracting distinguishing phrases.

This is the "VLM-generated" alternative to the hand-authored vocabulary
in `describer.py`. The goal is to let the model itself discover what
visual features are worth asking about, rather than hard-coding them.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
)


# Specific visual questions asked of each truck via BLIP-VQA.
# The goal is to get a short categorical answer for each.
VQA_QUESTIONS = [
    "What shape are the headlights?",
    "How many headlights does it have?",
    "What color is the bumper?",
    "What material is the bumper?",
    "What style is the grille?",
    "How many bars does the grille have?",
    "Is the grille chrome?",
    "Is the hood flat or curved?",
    "Does it have rounded or square fenders?",
    "Is the front end angular or rounded?",
    "Does the hood have lettering?",
    "Are the fenders separate from the body?",
    "Is there a chrome trim?",
    "Does it have turn signals in the bumper?",
    "Is the grille horizontal or vertical?",
    "What decade does this truck look like?",
]

# Prompts we'll ask BLIP about each truck.
# BLIP answers conditioned on an image + prompt prefix.
CAPTION_PROMPTS = [
    "a photo of the front of a pickup truck with",
    "the grille of this truck is",
    "the headlights of this truck are",
    "the bumper of this truck is",
    "the hood of this truck is",
    "the front end of this truck looks",
]


def load_blip(device: str = "cpu"):
    """Load BLIP captioning model (base, ~1GB)."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return model, processor


def load_blip_vqa(device: str = "cpu"):
    """Load BLIP visual question answering model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    model.eval()
    return model, processor


def vqa_answer(image_path: Path, question: str, model, processor, device: str = "cpu") -> str:
    """Ask BLIP-VQA a question about an image."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()


def generate_vqa_vocabulary(
    manifest: List[dict],
    device: str = "cpu",
    min_count: int = 2,
) -> Tuple[List[Tuple[str, str]], dict]:
    """Use BLIP-VQA to answer specific visual questions about each truck,
    then turn the distinguishing answers into vocabulary entries.
    """
    print("Loading BLIP-VQA...")
    model, processor = load_blip_vqa(device)

    all_answers = {}  # (question, answer) -> count
    per_image = {}

    print(f"Asking {len(VQA_QUESTIONS)} questions about {len(manifest)} images...")
    for i, entry in enumerate(manifest):
        path = Path(entry["processed_path"])
        img_facts = []
        for q in VQA_QUESTIONS:
            a = vqa_answer(path, q, model, processor, device)
            img_facts.append((q, a))
            key = (q, a)
            all_answers[key] = all_answers.get(key, 0) + 1
        per_image[str(path)] = img_facts
        print(f"  [{i+1}/{len(manifest)}] {entry['label']}: done")

    # Keep (question, answer) pairs where the answer appears on 2-N-1 trucks
    n = len(manifest)
    useful = [
        (q, a, count) for (q, a), count in all_answers.items()
        if min_count <= count < n
    ]
    useful.sort(key=lambda x: -x[2])
    print(f"\n{len(useful)} distinguishing (question, answer) pairs")

    # Turn each (question, answer) into a feature phrase
    # e.g., ("What shape are the headlights?", "round") -> "round headlights"
    vocabulary = []
    seen = set()
    for q, a, count in useful:
        # Construct a natural-sounding feature phrase from the q+a
        phrase = phrase_from_qa(q, a)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        opposite = phrase_opposite_from_qa(q, a)
        vocabulary.append((phrase, opposite))

    return vocabulary, per_image


def phrase_from_qa(question: str, answer: str) -> str:
    """Turn a VQA (question, answer) pair into a feature description."""
    q = question.lower()
    a = answer.strip(" .?!").lower()
    if not a or a in ("yes", "no", "unknown", "i don't know"):
        # Binary answers: convert to presence/absence phrases
        if "chrome" in q and a == "yes":
            return "a chrome bumper"
        if "chrome" in q and a == "no":
            return "a non-chrome bumper"
        if "lettering" in q and a == "yes":
            return "lettering on the hood"
        if "turn signals" in q and a == "yes":
            return "turn signals in the bumper"
        if "separate" in q and a == "yes":
            return "separate fenders from the body"
        return ""

    if "shape" in q and "headlight" in q:
        return f"{a} headlights"
    if "how many" in q and "headlight" in q:
        return f"{a} headlights"
    if "how many" in q and "grille" in q:
        return f"a grille with {a} bars"
    if "color" in q and "bumper" in q:
        return f"a {a} bumper"
    if "material" in q and "bumper" in q:
        return f"a {a} bumper"
    if "style" in q and "grille" in q:
        return f"a {a} grille"
    if "flat or curved" in q:
        return f"a {a} hood"
    if "rounded or square" in q:
        return f"{a} fenders"
    if "angular or rounded" in q:
        return f"{a} front end"
    if "horizontal or vertical" in q:
        return f"a {a} grille"
    if "decade" in q:
        return f"styling from the {a}"
    return ""


def phrase_opposite_from_qa(question: str, answer: str) -> str:
    """Generate a contrastive opposite phrase."""
    phrase = phrase_from_qa(question, answer)
    if not phrase:
        return ""
    # Simple negation — CLIP will score both
    if phrase.startswith("a "):
        return f"not {phrase}"
    return f"not {phrase}"


def caption_image(
    image_path: Path,
    prompt: str,
    model,
    processor,
    device: str = "cpu",
    max_tokens: int = 30,
) -> str:
    """Generate a caption for an image conditioned on a prompt prefix."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix if BLIP included it
    if caption.lower().startswith(prompt.lower()):
        caption = caption[len(prompt):].strip()
    return caption


def extract_phrases(caption: str) -> List[str]:
    """Pull meaningful noun/adjective phrases from a caption.

    Keeps phrases that contain a relevant anatomical keyword.
    """
    keywords = [
        "grille", "grill", "headlight", "headlights", "bumper", "hood",
        "fender", "fenders", "windshield", "chrome", "paint", "body",
        "front", "lights", "truck", "pickup",
    ]
    caption = caption.lower().strip()

    # Split into rough phrases on commas and "and"
    chunks = re.split(r",|\band\b|\bwith\b", caption)
    phrases = []
    for chunk in chunks:
        chunk = chunk.strip(" .?!;:")
        if not chunk or len(chunk) < 5 or len(chunk) > 80:
            continue
        if any(kw in chunk for kw in keywords):
            phrases.append(chunk)
    return phrases


def generate_vocabulary_from_images(
    manifest: List[dict],
    device: str = "cpu",
    min_count: int = 2,
) -> List[Tuple[str, str]]:
    """Run BLIP on every image with every prompt, collect phrases,
    keep the ones that appear on multiple trucks, and pair each with
    its negation to form a yes/no vocabulary.

    Returns a list of (feature, opposite) tuples in the same format
    as HUMAN_AUTHORED_VOCABULARY.
    """
    print(f"Loading BLIP model...")
    model, processor = load_blip(device)

    all_phrases = Counter()
    per_image_phrases = {}  # path -> set of phrases

    print(f"Captioning {len(manifest)} images with {len(CAPTION_PROMPTS)} prompts each...")
    for i, entry in enumerate(manifest):
        path = Path(entry["processed_path"])
        img_phrases = set()
        for prompt in CAPTION_PROMPTS:
            caption = caption_image(path, prompt, model, processor, device)
            phrases = extract_phrases(caption)
            img_phrases.update(phrases)
        per_image_phrases[str(path)] = img_phrases
        all_phrases.update(img_phrases)
        print(f"  [{i+1}/{len(manifest)}] {entry['label']}: {len(img_phrases)} phrases")

    # Keep phrases that appear on at least `min_count` trucks (otherwise they
    # are too idiosyncratic to be a useful key feature) and not ALL trucks
    # (otherwise they don't distinguish anything).
    n_images = len(manifest)
    useful = [
        (phrase, count) for phrase, count in all_phrases.most_common()
        if min_count <= count < n_images
    ]
    print(f"\nFound {len(useful)} distinguishing phrases (appear on {min_count}-{n_images - 1} trucks)")

    # Build (feature, opposite) pairs by negating each phrase
    vocabulary = []
    seen_keys = set()
    for phrase, count in useful:
        # Skip near-duplicates
        key = re.sub(r"\W+", "", phrase.lower())[:30]
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # The "opposite" is just the negation; CLIP will score both
        opposite = f"no {phrase}" if not phrase.startswith("no ") else phrase[3:]
        vocabulary.append((phrase, opposite))

    return vocabulary, per_image_phrases


if __name__ == "__main__":
    proc_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((proc_dir / "manifest.json").read_text())

    # Use VQA approach — much more effective than free-form captioning
    vocab, per_image = generate_vqa_vocabulary(manifest, min_count=2)

    print(f"\nVLM-generated vocabulary ({len(vocab)} entries):")
    for i, (feat, opp) in enumerate(vocab):
        print(f"  {i+1}. '{feat}'")

    # Save vocabulary and per-image answers
    out = {
        "vocabulary": vocab,
        "per_image_answers": {k: v for k, v in per_image.items()},
    }
    out_path = output_dir / "vlm_vocabulary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {out_path}")
