"""Render the trait-based dichotomous tree as SVG/PDF with the glossary."""

import json
from pathlib import Path

import graphviz


def render_trait_tree(tree_path: Path, glossary_path: Path, output_path: Path) -> Path:
    """Render the trait tree as graphviz SVG/PDF."""
    tree = json.loads(tree_path.read_text())
    glossary = json.loads(glossary_path.read_text())
    # Map trait_id -> top matching labels for tooltip/caption
    trait_info = {entry["trait_id"]: entry for entry in glossary}

    dot = graphviz.Digraph(
        comment="Ford F-Series Invented-Term Key",
        format="svg",
    )
    dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    counter = [0]

    def add_node(node):
        nid = str(counter[0])
        counter[0] += 1
        if node["type"] == "leaf":
            dot.node(
                nid,
                node["label"],
                fillcolor="#90EE90",
                shape="ellipse",
                style="filled",
                fontsize="11",
            )
        else:
            term = node["trait_name"]
            t_id = node["trait_id"]
            info = trait_info.get(t_id, {})
            top_labels = ", ".join(m["label"] for m in info.get("top_matches", [])[:3])
            label = f"{term}?\n(trait #{t_id})\nexemplars: {top_labels}"
            dot.node(
                nid,
                label,
                fillcolor="#FFE4B5",  # light amber — invented terms
                fontsize="9",
                fontname="Courier",
            )
            left_id = add_node(node["yes"])
            right_id = add_node(node["no"])
            dot.edge(nid, left_id, label="Yes")
            dot.edge(nid, right_id, label="No")
        return nid

    add_node(tree)

    svg_base = output_path.with_suffix("")
    dot.render(str(svg_base), format="svg", cleanup=True)
    dot.render(str(svg_base), format="pdf", cleanup=True)
    return Path(str(svg_base) + ".svg")


def print_tree_with_glossary(tree_path: Path, glossary_path: Path) -> None:
    """Pretty-print the tree to stdout with glossary entries."""
    tree = json.loads(tree_path.read_text())
    glossary = json.loads(glossary_path.read_text())
    trait_info = {entry["trait_id"]: entry for entry in glossary}

    # Collect used traits first
    used = set()
    def walk(n):
        if n["type"] == "decision":
            used.add(n["trait_id"])
            walk(n["yes"]); walk(n["no"])
    walk(tree)

    print("=" * 70)
    print("INVENTED-TERM DICHOTOMOUS TREE")
    print("=" * 70)
    print()

    def render(node, indent=""):
        if node["type"] == "leaf":
            print(f"{indent}-> {node['label']}")
            return
        term = node["trait_name"]
        t_id = node["trait_id"]
        print(f"{indent}[{term}] (trait #{t_id})?")
        print(f"{indent}|-- Yes:")
        render(node["yes"], indent + "|   ")
        print(f"{indent}`-- No:")
        render(node["no"], indent + "    ")

    render(tree)

    print()
    print("=" * 70)
    print(f"GLOSSARY — {len(used)} invented terms used by the tree")
    print("=" * 70)
    print()
    print("Each term is a cluster of CLIP patch embeddings. The 'exemplars' below")
    print("are the trucks whose patches match the cluster centroid most strongly —")
    print("this is the only definition the invented term has.")
    print()

    for t_id in sorted(used):
        info = trait_info[t_id]
        name = info["name"]
        matches = info["top_matches"]
        top_str = ", ".join(f"{m['label']} ({m['score']:.2f})" for m in matches[:4])
        print(f"  {name:<14} (trait #{t_id:2}) -> exemplars: {top_str}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent.parent / "outputs"
    tree_path = out_dir / "trait_tree.json"
    glossary_path = out_dir / "trait_glossary.json"

    print_tree_with_glossary(tree_path, glossary_path)

    svg = render_trait_tree(tree_path, glossary_path, out_dir / "trait_tree")
    print(f"\nSVG rendered to {svg}")
    print(f"PDF rendered to {svg.with_suffix('.pdf')}")
