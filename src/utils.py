"""Utilidades comunes de preprocesamiento, validación, cotas, reconstrucción de solución y despacho para BCP_k."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple
import networkx as nx

from src.license import configurar_licencia

__all__ = [
    "configurar_licencia",
    "preprocess_graph",
    "compute_graph_metrics",
    "compute_balance_bounds",
    "prepare_bcp_graph_data",
    "build_solution_summary",
    "solve_bcp",
]


def preprocess_graph(graph: nx.Graph) -> nx.Graph:
    """Extrae la componente conexa principal, limpia auto-lazos y re-indexa nodos a 1...n."""
    ug = nx.Graph(graph.to_undirected() if graph.is_directed() else graph)
    ug.remove_edges_from(nx.selfloop_edges(ug))

    if ug.number_of_nodes() < 2:
        raise ValueError("El grafo debe contener al menos 2 nodos.")

    if not nx.is_connected(ug):
        largest_cc = max(nx.connected_components(ug), key=len)
        ug = ug.subgraph(largest_cc).copy()

    clean = nx.convert_node_labels_to_integers(ug, first_label=1, ordering="sorted")
    clean.graph.update(graph.graph)
    return clean


def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """Calcula nodos (n), aristas (m), grado medio, densidad y diámetro."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    connected = nx.is_connected(graph) if n > 0 else False
    target = graph if connected else graph.subgraph(max(nx.connected_components(graph), key=len))

    return {
        "num_nodes": n,
        "num_edges": m,
        "avg_degree": round((2.0 * m) / n, 4) if n > 0 else 0.0,
        "density": round(float(nx.density(graph)), 4) if n > 1 else 0.0,
        "diameter": int(nx.diameter(target)) if target.number_of_nodes() > 1 else 0,
        "is_connected": connected,
    }


def compute_balance_bounds(n: int, k: int, epsilon: float = 0.0) -> Tuple[int, int]:
    """
    Calcula las cotas analíticas enteras inferior y superior para el balance de
    tamaño de cada partición bajo una tolerancia relativa epsilon en [0, 1).

        L_min(eps) = max{ 1, floor((1 - eps) * (n / k)) }
        U_max(eps) = ceil((1 + eps) * (n / k))

    Args:
        n: Número total de nodos (|V|).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa en [0, 1). Default: 0.0 (balance estricto).

    Returns:
        Tuple[int, int]: (L_min, U_max).
    """
    if k < 2:
        raise ValueError(f"El número de particiones k debe ser al menos 2, recibido: {k}")
    if n < k:
        raise ValueError(f"El número de nodos n ({n}) no puede ser menor que k ({k}).")
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(
            f"La tolerancia epsilon debe estar en el intervalo [0, 1), recibido: {epsilon}"
        )

    mean_capacity = n / k
    # Pequeña tolerancia numérica para evitar artefactos de punto flotante
    l_min = max(1, math.floor((1.0 - epsilon) * mean_capacity + 1e-9))
    u_max = math.ceil((1.0 + epsilon) * mean_capacity - 1e-9)

    return l_min, u_max


def prepare_bcp_graph_data(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
) -> Dict[str, Any]:
    """
    Valida el grafo de entrada y los parámetros (k, epsilon), construye el grafo
    interno indexado 1..n con aristas normalizadas (i < j) sin duplicados ni lazos,
    y calcula las cotas analíticas de balance (l_min, u_max).
    """
    if graph.is_directed():
        raise ValueError("El problema BCP_k requiere un grafo no dirigido.")
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise ValueError("El grafo de entrada debe ser no vacío y conexo.")

    orig_nodes = list(graph.nodes())
    n = len(orig_nodes)

    if k < 2:
        raise ValueError(f"El número de particiones k ({k}) debe ser mayor o igual a 2.")
    if n < k:
        raise ValueError(f"El número de nodos ({n}) debe ser mayor o igual a k ({k}).")

    node_to_idx = {node: idx + 1 for idx, node in enumerate(orig_nodes)}
    idx_to_node = {idx + 1: node for idx, node in enumerate(orig_nodes)}

    internal_graph = nx.Graph()
    internal_nodes = list(range(1, n + 1))
    internal_graph.add_nodes_from(internal_nodes)

    seen_edges: Set[Tuple[int, int]] = set()
    internal_edges: List[Tuple[int, int]] = []
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) not in seen_edges:
            seen_edges.add((i, j))
            internal_edges.append((i, j))
    internal_graph.add_edges_from(internal_edges)

    l_min, u_max = compute_balance_bounds(n, k, epsilon)
    partitions = list(range(1, k + 1))

    return {
        "n": n,
        "k": k,
        "epsilon": epsilon,
        "l_min": l_min,
        "u_max": u_max,
        "partitions": partitions,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "internal_graph": internal_graph,
        "internal_nodes": internal_nodes,
        "internal_edges": internal_edges,
    }


def build_solution_summary(
    graph: nx.Graph,
    meta: Mapping[str, Any],
    is_x_active: Optional[Callable[[int, int], bool]] = None,
    is_y_active: Optional[Callable[[int, int], bool]] = None,
) -> Dict[str, Any]:
    """
    Reconstruye las particiones con etiquetas originales, asignación nodo->bloque,
    lista de aristas cortadas, tamaños por bloque y verificación de conexidad inducida.
    Si `is_x_active` o `is_y_active` son None (sin solución factible), devuelve estructuras vacías.
    """
    k: int = int(meta["k"])
    partitions_dict: Dict[int, List[Any]] = {p: [] for p in range(1, k + 1)}
    assignment_dict: Dict[Any, int] = {}
    cut_edges_list: List[Tuple[Any, Any]] = []
    is_connected = False
    balance_sizes: Dict[int, int] = {p: 0 for p in range(1, k + 1)}

    if is_x_active is not None and is_y_active is not None:
        idx_to_node = meta["idx_to_node"]

        for i in meta["internal_nodes"]:
            orig_node = idx_to_node[i]
            assigned_p = None
            for p in range(1, k + 1):
                if is_x_active(i, p):
                    assigned_p = p
                    break
            if assigned_p is not None:
                partitions_dict[assigned_p].append(orig_node)
                assignment_dict[orig_node] = assigned_p

        for i, j in meta["internal_edges"]:
            if is_y_active(i, j):
                cut_edges_list.append((idx_to_node[i], idx_to_node[j]))

        connected_checks = []
        for p, members in partitions_dict.items():
            balance_sizes[p] = len(members)
            if len(members) == 0:
                connected_checks.append(False)
            else:
                connected_checks.append(nx.is_connected(graph.subgraph(members)))

        is_connected = all(connected_checks)

    return {
        "partitions": partitions_dict,
        "assignment": assignment_dict,
        "cut_edges": cut_edges_list,
        "is_connected": is_connected,
        "balance_stats": {
            "sizes": balance_sizes,
            "l_min": meta["l_min"],
            "u_max": meta["u_max"],
            "epsilon": meta["epsilon"],
        },
    }


def solve_bcp(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    time_limit: Optional[float] = None,
    solver: str = "gurobi",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Interfaz unificada para resolver el problema BCP_k con el solver especificado
    ('gurobi' o 'cplex').
    """
    name = solver.strip().lower()
    if name == "gurobi":
        from src.model_gurobi import solve_bcp_gurobi

        return solve_bcp_gurobi(
            graph=graph,
            k=k,
            epsilon=epsilon,
            time_limit=time_limit,
            **kwargs,
        )
    if name == "cplex":
        from src.model_cplex import solve_bcp_cplex

        return solve_bcp_cplex(
            graph=graph,
            k=k,
            epsilon=epsilon,
            time_limit=time_limit,
            **kwargs,
        )
    raise ValueError(f"Solver no soportado: {solver!r}. Use 'gurobi' o 'cplex'.")
