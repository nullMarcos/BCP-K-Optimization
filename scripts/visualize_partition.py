"""
Resuelve y visualiza particiones de BCP_k, exportando:
  1. Un archivo `.graphml` anotado con los atributos `partition`, `block_size` (en nodos)
     e `is_cut_edge` (en aristas) listo para colorear directamente en Gephi / Cytoscape / yEd.
  2. Una figura `.png` de alta resolución en `outputs/figures/` con los bloques coloreados
     y las aristas cortadas destacadas (soporta modo individual, comparación Con R4 vs Sin R4
     y comparación entre distintos valores de tolerancia epsilon).

Ejemplos de uso:
    # 1. Partición individual (genera PNG + GraphML anotado)
    .venv/bin/python scripts/visualize_partition.py --instance karate_club --k 2 --epsilon 0.05

    # 2. Comparar visualmente el efecto de la conectividad (Con R4 vs Sin R4)
    .venv/bin/python scripts/visualize_partition.py --instance karate_club --k 3 --epsilon 0.05 --compare-r4

    # 3. Comparar el efecto de relajar el balance (ej. eps=0.00 vs eps=0.15)
    .venv/bin/python scripts/visualize_partition.py --instance karate_club --k 2 --compare-epsilons 0.00 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from experiments.generate_profiles import (  # noqa: E402
    CATEGORICAL,
    FIG_DIR,
    TEXT_SECONDARY,
    apply_style,
)
from src.instance_loader import load_instances  # noqa: E402
from src.utils import solve_bcp  # noqa: E402

PARTITIONS_GRAPHML_DIR = PROJECT_ROOT / "data" / "graphml" / "partitions"
EXTENDED_PALETTE = CATEGORICAL + ["#9467bd", "#8c564b", "#e377c2", "#17becf"]


def compute_node_positions(graph: nx.Graph, seed: int = 42) -> Dict[Any, Any]:
    """
    Calcula coordenadas 2D deterministas para dibujar el grafo.
    Usa Kamada-Kawai en mallas planares medianas/pequeñas para preservar la geometría
    y Fruchterman-Reingold (spring_layout) con semilla fija en redes sociales/aleatorias.
    """
    family = graph.graph.get("family", "")
    if family == "mesh_planar" and graph.number_of_nodes() <= 150:
        try:
            return nx.kamada_kawai_layout(graph)
        except Exception:
            pass
    return nx.spring_layout(graph, seed=seed, k=1.8 / (graph.number_of_nodes() ** 0.5))


def export_annotated_graphml(
    graph: nx.Graph,
    res: Mapping[str, Any],
    out_path: Path,
) -> Path:
    """
    Guarda una copia del grafo en formato `.graphml` enriquecida con la solución:
      - Atributos de nodo: `partition` (int 1..k), `block_label` (str), `block_size` (int)
      - Atributos de arista: `is_cut_edge` (int 0/1), `edge_type` ('cut' o 'internal')
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = graph.copy()
    assignment: Mapping[Any, int] = res.get("assignment") or {}
    sizes: Mapping[int, int] = (res.get("balance_stats") or {}).get("sizes") or {}
    cut_set = {
        (u, v) if u <= v else (v, u)
        for u, v in (res.get("cut_edges") or [])
    }

    for node in annotated.nodes():
        p = int(assignment.get(node, 0))
        annotated.nodes[node]["partition"] = p
        annotated.nodes[node]["block_label"] = f"Bloque {p}" if p > 0 else "Sin asignar"
        annotated.nodes[node]["block_size"] = int(sizes.get(p, 0))

    for u, v in annotated.edges():
        edge_key = (u, v) if u <= v else (v, u)
        is_cut = 1 if edge_key in cut_set else 0
        annotated.edges[u, v]["is_cut_edge"] = is_cut
        annotated.edges[u, v]["edge_type"] = "cut" if is_cut else "internal"

    nx.write_graphml(annotated, out_path)
    return out_path


def _draw_partition_panel(
    ax: Any,
    graph: nx.Graph,
    pos: Mapping[Any, Any],
    res: Mapping[str, Any],
    title: str,
    with_labels: bool = True,
) -> None:
    """Dibuja un grafo particionado en un eje de Matplotlib."""
    ax.set_axis_off()
    if not res.get("has_solution"):
        ax.text(
            0.5,
            0.5,
            f"{title}\nEstado: {res.get('status')} (sin solución factible)",
            ha="center",
            va="center",
            fontsize=10,
            color=TEXT_SECONDARY,
        )
        return

    assignment: Mapping[Any, int] = res.get("assignment") or {}
    partitions: Mapping[int, List[Any]] = res.get("partitions") or {}
    bal = res.get("balance_stats") or {}
    sizes_map: Mapping[int, int] = bal.get("sizes") or {}
    cut_set = {
        (u, v) if u <= v else (v, u)
        for u, v in (res.get("cut_edges") or [])
    }

    internal_edges = []
    internal_colors = []
    cut_edges = []

    for u, v in graph.edges():
        edge_key = (u, v) if u <= v else (v, u)
        if edge_key in cut_set or assignment.get(u) != assignment.get(v):
            cut_edges.append((u, v))
        else:
            internal_edges.append((u, v))
            p = int(assignment.get(u, 1))
            internal_colors.append(EXTENDED_PALETTE[(p - 1) % len(EXTENDED_PALETTE)])

    # 1. Aristas cortadas (punteadas en gris/rojo suave al fondo)
    if cut_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=cut_edges,
            ax=ax,
            style=(0, (3, 3)),
            edge_color="#b0aba3",
            width=1.1,
            alpha=0.65,
        )

    # 2. Aristas internas de cada bloque (sólidas y del color de su partición)
    if internal_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=internal_edges,
            ax=ax,
            edge_color=internal_colors,
            width=1.8,
            alpha=0.85,
        )

    # 3. Nodos por bloque
    n = graph.number_of_nodes()
    node_size = 240 if n <= 50 else (140 if n <= 120 else 80)
    font_size = 7 if n <= 50 else 5

    for p in sorted(partitions):
        members = partitions[p]
        if not members:
            continue
        color = EXTENDED_PALETTE[(p - 1) % len(EXTENDED_PALETTE)]
        sub_conn = nx.is_connected(graph.subgraph(members))
        conn_tag = "" if sub_conn else f" [{nx.number_connected_components(graph.subgraph(members))} comp.!]"
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=members,
            ax=ax,
            node_color=color,
            edgecolors="#ffffff" if sub_conn else "#c01c28",
            linewidths=1.2 if sub_conn else 2.0,
            node_size=node_size,
            label=f"Bloque {p} (|V_{p}|={len(members)}){conn_tag}",
        )

    if with_labels and n <= 80:
        nx.draw_networkx_labels(
            graph,
            pos,
            ax=ax,
            font_size=font_size,
            font_color="#ffffff",
            font_weight="bold",
        )

    sizes_list = [sizes_map[p] for p in sorted(sizes_map)]
    conn_str = "Conexo" if res.get("is_connected") else "DESCONECTADO"
    subtitle = (
        f"Z* = {res.get('objective')} aristas cortadas | Tamaños: {sizes_list} "
        f"(cotas [{bal.get('l_min')}, {bal.get('u_max')}])\n"
        f"{conn_str} | Cortes R4: {res.get('lazy_cuts_added', 0)} | t = {res.get('runtime', 0.0):.3f}s"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=min(4, max(1, len(partitions))), fontsize=8)


def visualize_single(
    instance_name: str,
    graph: nx.Graph,
    k: int,
    epsilon: float,
    solver: str,
    time_limit: float,
    threads: Optional[int],
    use_lazy_connectivity: bool = True,
) -> Dict[str, Path]:
    """Resuelve una configuración individual y exporta el `.graphml` anotado y la figura `.png`."""
    res = solve_bcp(
        graph=graph,
        k=k,
        epsilon=epsilon,
        time_limit=time_limit,
        solver=solver,
        threads=threads,
        use_lazy_connectivity=use_lazy_connectivity,
        verbose=False,
    )
    if res.get("model") is not None and hasattr(res["model"], "dispose"):
        res["model"].dispose()

    r4_tag = "" if use_lazy_connectivity else "_noR4"
    stem = f"{instance_name}_k{k}_eps{int(round(epsilon * 100)):03d}_{solver.lower()}{r4_tag}"
    graphml_path = export_annotated_graphml(graph, res, PARTITIONS_GRAPHML_DIR / f"{stem}.graphml")

    pos = compute_node_positions(graph)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _draw_partition_panel(
        ax,
        graph,
        pos,
        res,
        title=f"{instance_name} (n={graph.number_of_nodes()}, m={graph.number_of_edges()}) — k={k}, ε={epsilon:.2f} [{solver.upper()}]",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"partition_{stem}.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return {"graphml": graphml_path, "figure": fig_path}


def visualize_compare_r4(
    instance_name: str,
    graph: nx.Graph,
    k: int,
    epsilon: float,
    solver: str,
    time_limit: float,
    threads: Optional[int],
) -> Dict[str, Path]:
    """Compara lado a lado la partición Con R4 (conexa) vs Sin R4 (relajación sin conexidad)."""
    pos = compute_node_positions(graph)
    fig, (ax_off, ax_on) = plt.subplots(1, 2, figsize=(13.2, 6.0))
    outputs: Dict[str, Path] = {}

    for ax, use_r4, label, tag in (
        (ax_off, False, "(a) Sin R4 (Conectividad desactivada)", "sin_R4"),
        (ax_on, True, "(b) Con R4 (Lazy Constraints de conectividad)", "con_R4"),
    ):
        res = solve_bcp(
            graph=graph,
            k=k,
            epsilon=epsilon,
            time_limit=time_limit,
            solver=solver,
            threads=threads,
            use_lazy_connectivity=use_r4,
            verbose=False,
        )
        if res.get("model") is not None and hasattr(res["model"], "dispose"):
            res["model"].dispose()

        stem = f"{instance_name}_k{k}_eps{int(round(epsilon * 100)):03d}_{tag}"
        outputs[f"graphml_{tag}"] = export_annotated_graphml(
            graph, res, PARTITIONS_GRAPHML_DIR / f"{stem}.graphml"
        )
        _draw_partition_panel(ax, graph, pos, res, title=label)

    fig.suptitle(
        f"Efecto de la restricción de conectividad (R4) en {instance_name} (k={k}, ε={epsilon:.2f}, {solver.upper()})",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"partition_compare_r4_{instance_name}_k{k}_eps{int(round(epsilon * 100)):03d}.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    outputs["figure"] = fig_path
    return outputs


def visualize_compare_epsilons(
    instance_name: str,
    graph: nx.Graph,
    k: int,
    epsilons: Sequence[float],
    solver: str,
    time_limit: float,
    threads: Optional[int],
) -> Dict[str, Path]:
    """Compara lado a lado las particiones óptimas bajo distintas tolerancias de balance ε."""
    pos = compute_node_positions(graph)
    fig, axes = plt.subplots(1, len(epsilons), figsize=(6.4 * len(epsilons), 6.0), squeeze=False)
    outputs: Dict[str, Path] = {}

    for idx, eps in enumerate(epsilons):
        res = solve_bcp(
            graph=graph,
            k=k,
            epsilon=eps,
            time_limit=time_limit,
            solver=solver,
            threads=threads,
            verbose=False,
        )
        if res.get("model") is not None and hasattr(res["model"], "dispose"):
            res["model"].dispose()

        eps_tag = f"eps{int(round(eps * 100)):03d}"
        stem = f"{instance_name}_k{k}_{eps_tag}_{solver.lower()}"
        outputs[f"graphml_{eps_tag}"] = export_annotated_graphml(
            graph, res, PARTITIONS_GRAPHML_DIR / f"{stem}.graphml"
        )
        _draw_partition_panel(
            axes[0][idx],
            graph,
            pos,
            res,
            title=f"({chr(97 + idx)}) Tolerancia ε = {eps:.2f}",
        )

    fig.suptitle(
        f"Efecto de la tolerancia de balance ε en {instance_name} (k={k}, {solver.upper()})",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eps_joined = "_vs_".join(f"{int(round(e * 100)):03d}" for e in epsilons)
    fig_path = FIG_DIR / f"partition_compare_eps_{instance_name}_k{k}_{eps_joined}.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    outputs["figure"] = fig_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resuelve, visualiza y exporta a GraphML particiones de BCP_k."
    )
    parser.add_argument("--instance", default="karate_club", help="Nombre de la instancia en data/processed")
    parser.add_argument("--k", type=int, default=2, help="Número de particiones k")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Tolerancia de balance ε")
    parser.add_argument("--solver", default="gurobi", choices=["gurobi", "cplex"], help="Solver a utilizar")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Límite de tiempo en segundos")
    parser.add_argument("--threads", type=int, default=None, help="Número máximo de hilos")
    parser.add_argument(
        "--compare-r4",
        action="store_true",
        help="Genera comparación visual lado a lado: Sin R4 vs Con R4.",
    )
    parser.add_argument(
        "--compare-epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Genera comparación visual entre múltiples valores de ε (ej. --compare-epsilons 0.00 0.15).",
    )
    args = parser.parse_args()

    apply_style()
    instances = load_instances(names=[args.instance])
    if args.instance not in instances:
        raise ValueError(f"Instancia no encontrada: {args.instance}")
    graph = instances[args.instance]

    if args.compare_r4:
        outputs = visualize_compare_r4(
            args.instance, graph, args.k, args.epsilon, args.solver, args.time_limit, args.threads
        )
    elif args.compare_epsilons:
        outputs = visualize_compare_epsilons(
            args.instance, graph, args.k, args.compare_epsilons, args.solver, args.time_limit, args.threads
        )
    else:
        outputs = visualize_single(
            args.instance, graph, args.k, args.epsilon, args.solver, args.time_limit, args.threads
        )

    for key, path in outputs.items():
        print(f"  {key:16s} -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
