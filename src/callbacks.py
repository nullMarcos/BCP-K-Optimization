"""
Módulo de Callbacks y Separadores Nodales para BCP_k.

Responsabilidades:
    - Lógica de separación de cortes de conectividad (R4).
    - Métodos BFS/DFS para identificación de componentes conexas inducidas.
    - Poda de separadores nodales a separadores minimales (find_minimal_separator).
    - Separador común para ambos solvers (find_connectivity_cuts).
    - Callbacks de Lazy Constraints para Gurobi (bcp_connectivity_callback)
      y soporte reusable para el callback equivalente en CPLEX (model_cplex.py).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple
import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    gp = None
    GRB = None
    _HAS_GUROBI = False


def find_minimal_separator(
    graph: nx.Graph,
    u: Any,
    v: Any,
    candidate_separator: Set[Any],
) -> Set[Any]:
    """
    Poda un separador nodal candidato a uno minimal S <= candidate_separator
    que desconecta el par de nodos (u, v) en graph.

    Esta rutina es independiente del solver y puede ser reutilizada tanto por
    el modelo de Gurobi como por el de CPLEX.

    Args:
        graph: Grafo no dirigido.
        u: Nodo origen.
        v: Nodo destino.
        candidate_separator: Conjunto de nodos que desconectan u y v en graph.

    Returns:
        Set[Any]: Subconjunto minimal que desconecta u y v.
    """
    s_min = set(candidate_separator)
    # Grafo auxiliar sin el separador candidato completo
    aux_graph = graph.copy()
    aux_graph.remove_nodes_from(s_min)

    # Si por alguna razón u y v ya estuvieran conectados, retornamos el candidato completo
    if nx.has_path(aux_graph, u, v):
        return set(candidate_separator)

    # Intentamos remover cada nodo del separador para verificar si sigue siendo necesario
    for s in list(s_min):
        nbrs_in_aux = [nbr for nbr in graph.neighbors(s) if nbr in aux_graph]
        aux_graph.add_node(s)
        aux_graph.add_edges_from((s, nbr) for nbr in nbrs_in_aux)

        if nx.has_path(aux_graph, u, v):
            # El nodo s es estrictamente necesario para mantener desconectados u y v
            aux_graph.remove_node(s)
        else:
            # El nodo s era redundante para desconectar u y v
            s_min.remove(s)

    return s_min


def find_connectivity_cuts(
    internal_graph: nx.Graph,
    partitions_nodes: Dict[int, List[int]],
    deduplicate: bool = False,
) -> List[Tuple[int, int, int, Set[int]]]:
    """
    Busca los cortes de conectividad (R4) violados por una asignación de nodos.

    Para cada partición p se revisa si G[V_p] es conexo. Si no lo es, se agrega
    un corte por cada componente C (incluida la más grande), separando un nodo
    u de C de un nodo v de V_p fuera de C, con la vecindad N(C) podada con
    find_minimal_separator (queda minimal por inclusión, no necesariamente de
    cardinalidad mínima).

    Si `deduplicate=True`, evita recalcular e inyectar dos veces el mismo corte
    simétrico (p, {u, v}, S) cuando dos componentes comparten la misma vecindad
    o producen el mismo separador minimal S.

    No usa nada del solver, por lo que sirve tanto para Gurobi como para CPLEX.

    Args:
        internal_graph: Grafo con los nodos indexados 1..n.
        partitions_nodes: Diccionario {p: [nodos asignados a p]}.
        deduplicate: Si es True, filtra cortes simétricos repetidos (u, v, S) y (v, u, S).

    Returns:
        Lista de tuplas (p, u, v, S), cada una representa el corte
        x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}.
    """
    cortes: List[Tuple[int, int, int, Set[int]]] = []
    seen_neighborhoods: Set[Tuple[int, int, int, frozenset[int]]] = set()
    seen_cuts: Set[Tuple[int, int, int, frozenset[int]]] = set()

    for p, v_p in partitions_nodes.items():
        if len(v_p) <= 1:
            continue

        componentes = list(nx.connected_components(internal_graph.subgraph(v_p)))
        if len(componentes) <= 1:
            continue

        componentes.sort(key=len)

        for comp in componentes:
            u_rep = next(iter(comp))
            v_rep = next(w for w in v_p if w not in comp)
            # N(C): ninguno de estos nodos está en V_p, si no C no sería componente
            vecindad = {
                w for nodo in comp for w in internal_graph.neighbors(nodo)
                if w not in comp
            }
            pair_u, pair_v = (u_rep, v_rep) if u_rep <= v_rep else (v_rep, u_rep)

            if deduplicate:
                nbr_key = (p, pair_u, pair_v, frozenset(vecindad))
                if nbr_key in seen_neighborhoods:
                    continue
                seen_neighborhoods.add(nbr_key)

            separador = find_minimal_separator(internal_graph, u_rep, v_rep, vecindad)

            if deduplicate:
                cut_key = (p, pair_u, pair_v, frozenset(separador))
                if cut_key in seen_cuts:
                    continue
                seen_cuts.add(cut_key)

            cortes.append((p, u_rep, v_rep, separador))

    return cortes


def bcp_connectivity_callback(model: Any, where: int) -> None:
    """
    Callback de Gurobi (MIPSOL) para agregar Lazy Constraints de conectividad (R4).

    Verifica si los subgrafos inducidos G[V_p] son conexos en la solución entera actual.
    Si se detectan particiones desconectadas, identifica componentes conexas mediante
    BFS/DFS, calcula separadores nodales mínimos y añade los cortes violados:
        x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}

    Args:
        model: Instancia del modelo Gurobi en optimización.
        where: Código de fase de Gurobi (GRB.Callback.*).
    """
    if not _HAS_GUROBI:
        raise ImportError(
            "gurobipy no está disponible en este entorno. "
            "Se requiere gurobipy para ejecutar bcp_connectivity_callback."
        )

    if where != GRB.Callback.MIPSOL:
        return

    data = model._bcp_data
    internal_graph: nx.Graph = data["internal_graph"]
    k: int = data["k"]
    x = data["x"]
    internal_nodes: List[int] = data["internal_nodes"]

    inicio = time.perf_counter()

    # Valores de la solución entera candidata
    x_sol = model.cbGetSolution(x)

    partitions_nodes = {
        p: [i for i in internal_nodes if x_sol[i, p] > 0.5]
        for p in range(1, k + 1)
    }

    for p, u_rep, v_rep, separador in find_connectivity_cuts(
        internal_graph, partitions_nodes, deduplicate=True
    ):
        # R4: x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}
        model.cbLazy(
            x[u_rep, p] + x[v_rep, p] - 1
            <= gp.quicksum(x[s, p] for s in separador)
        )
        data["lazy_cuts_count"] += 1

    # tiempo acumulado dentro del callback, para las estadísticas del informe
    data["callback_time"] = data.get("callback_time", 0.0) + time.perf_counter() - inicio
