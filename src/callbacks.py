"""
Módulo de Callbacks y Separadores Nodales para BCP_k.

Asignado a: Sebastián
Responsabilidades:
    - Lógica de separación de cortes de conectividad (R4).
    - Métodos BFS/DFS para identificación de componentes conexas inducidas.
    - Determinación y poda de separadores nodales mínimos (find_minimal_separator).
    - Callbacks de Lazy Constraints para Gurobi (bcp_connectivity_callback)
      y soporte reusable para el callback equivalente en CPLEX (model_cplex.py).
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple
import networkx as nx
import gurobipy as gp
from gurobipy import GRB


def find_minimal_separator(
    graph: nx.Graph,
    u: Any,
    v: Any,
    candidate_separator: Set[Any],
) -> Set[Any]:
    """
    Poda un separador nodal candidato a un separador nodal mínimo S <= candidate_separator
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


def bcp_connectivity_callback(model: gp.Model, where: int) -> None:
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
    if where != GRB.Callback.MIPSOL:
        return

    data = model._bcp_data
    internal_graph: nx.Graph = data["internal_graph"]
    k: int = data["k"]
    x = data["x"]
    internal_nodes: List[int] = data["internal_nodes"]

    # Obtener valores de la solución entera actual
    x_sol = model.cbGetSolution(x)

    # Analizar la conexidad de cada partición p
    for p in range(1, k + 1):
        v_p = [i for i in internal_nodes if x_sol[i, p] > 0.5]
        if len(v_p) <= 1:
            continue

        sub_g = internal_graph.subgraph(v_p)
        components = list(nx.connected_components(sub_g))

        if len(components) <= 1:
            continue

        # La partición p está desconectada en len(components) componentes
        # Ordenamos las componentes por tamaño ascendente
        components.sort(key=len)

        # Para cada componente aislada c_comp
        for c_comp in components:
            # Vecindad abierta en internal_graph de c_comp hacia el exterior
            n_c = set()
            for u in c_comp:
                for nbr in internal_graph.neighbors(u):
                    if nbr not in c_comp:
                        n_c.add(nbr)

            # En la solución actual, ningún nodo de n_c pertenece a v_p
            # Tomamos un representante u de c_comp y un representante v fuera de c_comp
            other_nodes = [w for w in v_p if w not in c_comp]
            if not other_nodes:
                continue

            u_rep = next(iter(c_comp))
            v_rep = other_nodes[0]

            # Obtenemos un separador nodal mínimo S <= n_c que desconecta u_rep y v_rep
            s_separator = find_minimal_separator(internal_graph, u_rep, v_rep, n_c)
            if not s_separator:
                s_separator = n_c

            # Restricción perezosa R4: x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}
            model.cbLazy(
                x[u_rep, p] + x[v_rep, p] - 1
                <= gp.quicksum(x[s, p] for s in s_separator)
            )
            data["lazy_cuts_count"] += 1
