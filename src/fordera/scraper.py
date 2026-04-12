"""Scrape Ford F-series front-profile images from Street Trucks Mag."""

import json
import urllib.request
from pathlib import Path

# Image manifest: each entry maps a URL to its year label(s)
# Derived from https://www.streettrucksmag.com/complete-history-of-the-ford-f-series-pickup/
IMAGE_MANIFEST = [
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-01.webp", "years": [1948, 1949, 1950], "label": "1948-1950"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-03.webp", "years": [1951], "label": "1951"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-04.webp", "years": [1952], "label": "1952"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-05.webp", "years": [1953], "label": "1953"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-06.webp", "years": [1954], "label": "1954"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-07-1.webp", "years": [1955], "label": "1955"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-08.webp", "years": [1956], "label": "1956"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-10.webp", "years": [1957], "label": "1957"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-11.webp", "years": [1958], "label": "1958"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-12.webp", "years": [1959], "label": "1959"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-13.webp", "years": [1960], "label": "1960"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-16.webp", "years": [1961], "label": "1961"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-17.webp", "years": [1962], "label": "1962"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-18.webp", "years": [1963], "label": "1963"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-19.webp", "years": [1964], "label": "1964"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-20.webp", "years": [1965], "label": "1965"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-21.webp", "years": [1966], "label": "1966"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-24.webp", "years": [1967], "label": "1967"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-25.webp", "years": [1967], "label": "1967_alt"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-26.webp", "years": [1968], "label": "1968"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-27.webp", "years": [1968], "label": "1968_alt"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-28.webp", "years": [1969], "label": "1969"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-29.webp", "years": [1969], "label": "1969_alt"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-30.webp", "years": [1970], "label": "1970"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-31.webp", "years": [1970], "label": "1970_alt"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-33.webp", "years": [1971], "label": "1971"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-35.webp", "years": [1972], "label": "1972"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-38.webp", "years": [1973, 1974, 1975], "label": "1973-1975"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-40.webp", "years": [1976, 1977], "label": "1976-1977"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-42.webp", "years": [1978], "label": "1978"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-43.webp", "years": [1978], "label": "1978_alt"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-44.webp", "years": [1979], "label": "1979"},
    {"url": "https://www.streettrucksmag.com/wp-content/uploads/2023/09/F100-1812-FGEN-45.webp", "years": [1979], "label": "1979_alt"},
]


def scrape_images(output_dir: Path) -> list[dict]:
    """Download all images and return the manifest with local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for entry in IMAGE_MANIFEST:
        url = entry["url"]
        label = entry["label"]
        ext = Path(url).suffix  # .webp
        filename = f"{label}{ext}"
        filepath = output_dir / filename

        if not filepath.exists():
            print(f"Downloading {label}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                filepath.write_bytes(resp.read())
        else:
            print(f"Already have {label}")

        manifest.append({
            "label": label,
            "years": entry["years"],
            "path": str(filepath),
            "url": url,
        })

    return manifest


def save_manifest(manifest: list[dict], output_path: Path) -> None:
    """Save manifest as JSON."""
    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {output_path}")


if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    manifest = scrape_images(raw_dir)
    save_manifest(manifest, raw_dir / "manifest.json")
    print(f"\nDownloaded {len(manifest)} images")
    # Determine unique classes
    unique_labels = {e["label"].split("_")[0] for e in manifest}
    print(f"Unique class labels: {len(unique_labels)}")
    for label in sorted(unique_labels):
        count = sum(1 for e in manifest if e["label"].split("_")[0] == label)
        print(f"  {label}: {count} image(s)")
