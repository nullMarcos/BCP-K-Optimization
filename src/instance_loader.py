"""
Módulo Generador y Cargador de Instancias
Lee los datasets, genera las mallas regulares 2D y los grafos sintéticos con semilla fija,
y transforma todo a lista de adyacencia
"""

from __future__ import annotations

import gzip
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
from src.utils import compute_graph_metrics, preprocess_graph

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SEED = 42


def to_adjacency_list(graph: nx.Graph) -> Dict[int, List[int]]:
    """Transforma un grafo en un diccionario de lista de adyacencia."""
    return {int(u): sorted(int(v) for v in graph.neighbors(u)) for u in sorted(graph.nodes())}


def save_adjacency_list(graph: nx.Graph, path: Path) -> None:
    """Guarda el grafo como lista de adyacencia"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# family={graph.graph.get('family', 'unknown')}"]
    lines += [f"{u} {' '.join(map(str, nbrs))}".strip() for u, nbrs in to_adjacency_list(graph).items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_adjacency_list(path: Path) -> nx.Graph:
    """Lee un archivo `.adjlist` desde `data/processed/` y reconstruye el grafo."""
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    family = first_line.split("family=")[1].strip() if "family=" in first_line else "unknown"
    graph = nx.read_adjlist(path, nodetype=int, create_using=nx.Graph)
    graph.graph.update({"name": path.stem, "family": family})
    return graph


def parse_raw_file(path: Path, max_nodes: Optional[int] = 200) -> nx.Graph:
    """Parsea un archivo crudo y lo preprocesa"""
    raw = nx.Graph()
    is_mtx = ".mtx" in path.suffixes
    opener = gzip.open if path.suffix == ".gz" else open
    skip_dims = is_mtx

    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "%")):
                continue
            parts = line.replace(",", " ").split()
            if skip_dims and len(parts) >= 3:
                skip_dims = False
                continue
            if len(parts) >= 2 and parts[0] != parts[1]:
                raw.add_edge(parts[0], parts[1])

    graph = preprocess_graph(raw)
    if not is_mtx and max_nodes and graph.number_of_nodes() > max_nodes:
        hub = max(graph.nodes(), key=graph.degree)
        bfs_nodes = [hub] + [v for _, v in nx.bfs_edges(graph, source=hub)][: max_nodes - 1]
        graph = preprocess_graph(graph.subgraph(bfs_nodes))

    graph.graph.update({
        "name": path.name.split(".")[0],
        "family": "mesh_planar" if is_mtx else "social_scale_free",
    })
    return graph


def process_all_instances(
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    snap_max_nodes: Optional[int] = 200,
) -> Dict[str, nx.Graph]:
    """Transforma los datasets a listas de adyacencia"""
    raw_path, proc_path = Path(raw_dir), Path(processed_dir)
    """Grafos de la biblioteca networkx (sinteticos y propios de la biblioteca)"""
    builtin_specs = {
        "karate_club": (nx.karate_club_graph(), "social_scale_free"),
        "les_miserables": (nx.les_miserables_graph(), "social_scale_free"),
        "grid_10x10": (nx.grid_2d_graph(10, 10), "mesh_planar"),
        "grid_12x12": (nx.grid_2d_graph(12, 12), "mesh_planar"),
        "er_n100_p006": (nx.erdos_renyi_graph(100, 0.06, seed=DEFAULT_SEED), "synthetic_random"),
        "er_n150_p005": (nx.erdos_renyi_graph(150, 0.05, seed=DEFAULT_SEED), "synthetic_random"),
        "ba_n100_m2": (nx.barabasi_albert_graph(100, 2, seed=DEFAULT_SEED), "synthetic_random"),
        "ba_n150_m2": (nx.barabasi_albert_graph(150, 2, seed=DEFAULT_SEED), "synthetic_random"),
    }

    instances: Dict[str, nx.Graph] = {}
    for name, (raw_g, family) in builtin_specs.items():
        g = preprocess_graph(raw_g)
        g.graph.update({"name": name, "family": family})
        instances[name] = g

    if raw_path.is_dir():
        for file_path in sorted(raw_path.iterdir()):
            if file_path.is_file() and file_path.name != ".gitkeep":
                g = parse_raw_file(file_path, max_nodes=snap_max_nodes)
                instances[g.graph["name"]] = g

    for name, g in instances.items():
        save_adjacency_list(g, proc_path / f"{name}.adjlist")

    return instances


def load_instances(
    names: Optional[Iterable[str]] = None,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
) -> Dict[str, nx.Graph]:
    """Carga las instancias"""
    proc_path = Path(processed_dir)
    adj_files = sorted(proc_path.glob("*.adjlist")) if proc_path.is_dir() else []
    instances = (
        {f.stem: read_adjacency_list(f) for f in adj_files}
        if adj_files
        else process_all_instances(raw_dir=raw_dir, processed_dir=processed_dir)
    )
    if names:
        instances = {k: v for k, v in instances.items() if k in set(names)}
    return dict(sorted(instances.items(), key=lambda item: (item[1].number_of_nodes(), item[0])))


if __name__ == "__main__":
    loaded = process_all_instances()
    print(f"Instancias transformadas a lista de adyacencia en '{DEFAULT_PROCESSED_DIR}': {len(loaded)}")
    for name, g in loaded.items():
        m = compute_graph_metrics(g)
        print(
            f"  - {name:16s} [{g.graph.get('family', ''):17s}] | "
            f"n={m['num_nodes']:4d} | m={m['num_edges']:5d} | "
            f"grado_medio={m['avg_degree']:6.2f} | densidad={m['density']:6.4f} | diametro={m['diameter']}"
        )
