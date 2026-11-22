"""Utilidades de preprocesamiento y métricas topológicas para BCP_k."""

from __future__ import annotations

from typing import Any, Dict
import networkx as nx

from src.license import configurar_licencia

__all__ = ["configurar_licencia", "preprocess_graph", "compute_graph_metrics"]


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
