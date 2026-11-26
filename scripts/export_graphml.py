"""Exporta todos los grafos `.adjlist` de `data/processed/` a formato `.graphml` sin modificar ni borrar archivos existentes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
from src.instance_loader import DEFAULT_PROCESSED_DIR, read_adjacency_list

DEFAULT_GRAPHML_DIR = PROJECT_ROOT / "data" / "graphml"


def export_all_to_graphml(input_dir: Path, output_dir: Path) -> int:
    """Convierte todos los archivos `.adjlist` de `input_dir` a `.graphml` en `output_dir`."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {input_dir}")

    adjlist_files = sorted(input_dir.glob("*.adjlist"))
    if not adjlist_files:
        print(f"No se encontraron archivos .adjlist en {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    for adj_path in adjlist_files:
        graph = read_adjacency_list(adj_path)
        out_path = output_dir / f"{adj_path.stem}.graphml"
        nx.write_graphml(graph, out_path)
        print(
            f"[OK] {adj_path.name} -> {out_path.relative_to(PROJECT_ROOT)} "
            f"({graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas)"
        )
        exported += 1

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte los grafos .adjlist a formato .graphml para visualización."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directorio con archivos .adjlist (por defecto: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GRAPHML_DIR,
        help="Directorio de salida para los .graphml (por defecto: data/graphml)",
    )
    args = parser.parse_args()

    count = export_all_to_graphml(args.input_dir, args.output_dir)
    print(f"\nTotal exportados: {count} grafos en {args.output_dir}")


if __name__ == "__main__":
    main()
