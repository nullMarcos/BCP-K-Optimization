"""
Módulo del Modelo de Optimización BCP_k en Gurobi (Branch-and-Cut).

Este módulo implementa el modelo de Programación Lineal Entera (PLE) para resolver
el problema de Particionamiento Balanceado y Conexo de Grafos en k Grupos (BCP_k)
utilizando gurobipy, conforme a la formulación matemática documentada para el curso
de Optimización (UdeC):
    - Variables de asignación x_{i,p} y aristas cortadas y_{ij}.
    - R1: Asignación única por nodo.
    - R2: Balance paramétrico de tamaño ((1 + eps)-balance) con cotas L_min y U_max.
    - R3: Definición linealizada de aristas cortadas.
    - R4: Conectividad inducida mediante Lazy Constraints y separadores nodales mínimos.
    - R5: Dominio binario para variables.
    - R6: Ruptura de simetrías por precedencia dinámica y anclaje canónico.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB
import networkx as nx


from src.utils import configurar_licencia


def compute_balance_bounds(n: int, k: int, epsilon: float = 0.0) -> Tuple[int, int]:
    """
    Calcula las cotas analíticas entera inferior y superior para el balance
    de tamaño de cada partición bajo una tolerancia relativa epsilon en [0, 1).

    L_min(eps) = max{ 1, floor((1 - eps) * (n / k)) }
    U_max(eps) = ceil((1 + eps) * (n / k))

    Args:
        n: Número total de nodos (|V|).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa en [0, 1). Default: 0.0 (balance estricto).

    Returns:
        Tuple[int, int]: (L_min, U_max).
    """
    if k < 1:
        raise ValueError("El número de particiones k debe ser al menos 1.")
    if n < k:
        raise ValueError(f"El número de nodos n ({n}) no puede ser menor que k ({k}).")
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"La tolerancia epsilon debe estar en el intervalo [0, 1), recibido: {epsilon}")

    mean_capacity = n / k
    # Se añade un pequeño epsilon numérico para evitar artefactos de punto flotante
    l_min = max(1, math.floor((1.0 - epsilon) * mean_capacity + 1e-9))
    u_max = math.ceil((1.0 + epsilon) * mean_capacity - 1e-9)

    return l_min, u_max


def find_minimal_separator(
    graph: nx.Graph,
    u: Any,
    v: Any,
    candidate_separator: Set[Any],
) -> Set[Any]:
    """
    Poda un separador nodal candidato a un separador nodal mínimo S <= candidate_separator
    que desconecta el par (u, v) en graph.

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

    # Intentamos remover cada nodo del separador para ver si sigue siendo necesario
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
    Callback de Gurobi (MIPSOL) para agregar Lazy Constraints, (R4).
    Verifica si los subgrafos inducidos G[V_p] son conexos. Si se detectan particiones
    desconectadas, identifica separadores nodales mínimos y añade cortes violados.
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


def build_bcp_gurobi_model(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    model_name: str = "BCP_k_Gurobi",
    env: Optional[gp.Env] = None,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Construye la instancia del modelo de Programación Lineal Entera (PLE) en Gurobi.

    Args:
        graph: Grafo no dirigido y conexo (NetworkX).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa de balance en [0, 1).
        use_symmetry_breaking: Si es True, incluye restricciones de orden canónico (R6).
        use_lazy_connectivity: Si es True, habilita R4 vía Lazy Constraints Callback.
        model_name: Nombre asignado al modelo de Gurobi.
        env: Entorno opcional de Gurobi (gp.Env) previamente configurado con licencia.

    Returns:
        Tuple[gp.Model, Dict[str, Any]]:
            - model: Instancia de gp.Model configurada.
            - metadata: Diccionario con mappings de nodos, variables x, y, cotas y parámetros.
    """
    if graph.is_directed():
        raise ValueError("El problema BCP_k requiere un grafo no dirigido.")
    if not nx.is_connected(graph):
        raise ValueError("El grafo de entrada debe ser conexo.")

    orig_nodes = list(graph.nodes())
    n = len(orig_nodes)

    if k < 2:
        raise ValueError(f"El número de particiones k ({k}) debe ser mayor o igual a 2.")
    if n < k:
        raise ValueError(f"El número de nodos ({n}) debe ser mayor o igual a k ({k}).")

    # Crear diccionarios bidireccionales para mapear nodos <-> índices 1..n
    node_to_idx = {node: i + 1 for i, node in enumerate(orig_nodes)}
    idx_to_node = {i + 1: node for i, node in enumerate(orig_nodes)}

    # Grafo interno con nodos indexados 1..n
    internal_graph = nx.Graph()
    internal_graph.add_nodes_from(range(1, n + 1))
    internal_edges: List[Tuple[int, int]] = []
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        if i > j:
            i, j = j, i
        internal_edges.append((i, j))
    internal_graph.add_edges_from(internal_edges)

    # Cotas de balance (R2)
    l_min, u_max = compute_balance_bounds(n, k, epsilon)

    # Inicializar modelo Gurobi con el entorno especificado o el entorno por defecto
    if env is not None:
        model = gp.Model(model_name, env=env)
    else:
        model = gp.Model(model_name)

    # 1. Variables de Decisión (R5)
    # x[i, p]: asignación del nodo i a la partición p
    partitions = list(range(1, k + 1))
    internal_nodes = list(range(1, n + 1))

    x = model.addVars(
        internal_nodes,
        partitions,
        vtype=GRB.BINARY,
        name="x",
    )

    # y[i, j]: 1 si la arista (i, j) está cortada
    y = model.addVars(
        internal_edges,
        vtype=GRB.BINARY,
        name="y",
    )

    # 2. Función Objetivo (Ecuación 3)
    # Minimizar Z = sum_{(i,j) in E} y_{ij}
    model.setObjective(gp.quicksum(y[i, j] for (i, j) in internal_edges), GRB.MINIMIZE)

    # 3. Restricciones
    # (R1) Asignación Única: cada nodo está en exactamente una partición
    for i in internal_nodes:
        model.addConstr(
            gp.quicksum(x[i, p] for p in partitions) == 1,
            name=f"R1_assign_{i}",
        )

    # (R2) Balance de Tamaño Paramétrico ((1 + eps)-balance)
    for p in partitions:
        model.addConstr(
            gp.quicksum(x[i, p] for i in internal_nodes) >= l_min,
            name=f"R2_lower_{p}",
        )
        model.addConstr(
            gp.quicksum(x[i, p] for i in internal_nodes) <= u_max,
            name=f"R2_upper_{p}",
        )

    # (R3) Definición de Aristas Cortadas
    for i, j in internal_edges:
        for p in partitions:
            model.addConstr(
                y[i, j] >= x[i, p] - x[j, p],
                name=f"R3_cut_pos_{i}_{j}_{p}",
            )
            model.addConstr(
                y[i, j] >= x[j, p] - x[i, p],
                name=f"R3_cut_neg_{i}_{j}_{p}",
            )

    # (R6) Rompimiento de Simetrías (Symmetry Breaking)
    if use_symmetry_breaking:
        # Anclaje canónico del nodo 1 a la partición 1
        model.addConstr(x[1, 1] == 1, name="R6_anchor_first")

        # Ningún nodo con índice menor que p puede abrir la partición p
        for p in range(2, k + 1):
            for i in range(1, p):
                model.addConstr(x[i, p] == 0, name=f"R6_prefix_{i}_{p}")

        # Condición de precedencia dinámica: nodo i solo puede ir a p si algún j < i ya está en p-1
        for i in range(2, n + 1):
            for p in range(2, k + 1):
                model.addConstr(
                    gp.quicksum(x[j, p - 1] for j in range(1, i)) >= x[i, p],
                    name=f"R6_precedence_{i}_{p}",
                )

    # Parámetro de restricciones perezosas
    if use_lazy_connectivity:
        model.Params.LazyConstraints = 1

    metadata: Dict[str, Any] = {
        "x": x,
        "y": y,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "internal_graph": internal_graph,
        "internal_nodes": internal_nodes,
        "internal_edges": internal_edges,
        "k": k,
        "n": n,
        "epsilon": epsilon,
        "l_min": l_min,
        "u_max": u_max,
        "use_symmetry_breaking": use_symmetry_breaking,
        "use_lazy_connectivity": use_lazy_connectivity,
        "lazy_cuts_count": 0,
    }

    model._bcp_data = metadata
    return model, metadata


def solve_bcp_gurobi(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    log_file: Optional[str] = None,
    verbose: bool = True,
    env: Optional[gp.Env] = None,
    lic_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construye y resuelve el modelo BCP_k en Gurobi con callbacks perezosos.

    Args:
        graph: Grafo no dirigido y conexo (NetworkX).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa de balance en [0, 1).
        time_limit: Límite de tiempo en segundos (TimeLimit de Gurobi).
        mip_gap: Tolerancia relativa de brecha de optimalidad (MIPGap).
        threads: Número máximo de hilos paralelos a utilizar.
        use_symmetry_breaking: Si se activan las restricciones R6.
        use_lazy_connectivity: Si se activan las Lazy Constraints R4.
        log_file: Ruta opcional para guardar el log de salida de Gurobi.
        verbose: Si es False, silencia la salida estándar de Gurobi.
        env: Instancia de entorno gp.Env opcional.
        lic_path: Ruta opcional al archivo gurobi.lic para invocar configurar_licencia().

    Returns:
        Dict[str, Any]: Diccionario con los resultados de la optimización:
            - status: Código textual del estado del solver (OPTIMAL, TIME_LIMIT, etc.).
            - objective: Valor óptimo (número de aristas cortadas) o None.
            - best_bound: Cota dual (BestBound) reportada por el solver.
            - mip_gap: Brecha de optimalidad final (MIPGap).
            - runtime: Tiempo total de resolución en segundos.
            - node_count: Número de nodos explorados en el árbol Branch-and-Cut.
            - lazy_cuts_added: Cantidad de restricciones de conectividad agregadas dinámicamente.
            - partitions: Mapeo {p: [nodos]} con etiquetas originales del grafo.
            - assignment: Mapeo {nodo: p} para cada nodo del grafo.
            - cut_edges: Lista de tuplas (u, v) correspondientes a las aristas cortadas.
            - is_connected: Booleano que certifica si cada partición es 100% conexa.
            - balance_stats: Tamaños de cada bloque {p: tamaño} y rango [L_min, U_max].
    """
    if env is None:
        if lic_path is not None:
            configurar_licencia(ruta_licencia=lic_path, verbose=verbose)
        else:
            try:
                configurar_licencia(verbose=verbose)
            except (FileNotFoundError, ValueError):
                pass

    model, meta = build_bcp_gurobi_model(
        graph=graph,
        k=k,
        epsilon=epsilon,
        use_symmetry_breaking=use_symmetry_breaking,
        use_lazy_connectivity=use_lazy_connectivity,
        env=env,
    )

    # Parámetros de ejecución
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = max(0.0, float(time_limit))
    if mip_gap is not None:
        model.Params.MIPGap = max(0.0, float(mip_gap))
    if threads is not None:
        model.Params.Threads = max(1, int(threads))
    if log_file:
        model.Params.LogFile = str(log_file)

    # Registro del callback de conectividad perezosa
    callback_fn = bcp_connectivity_callback if use_lazy_connectivity else None

    start_wall_time = time.perf_counter()
    if callback_fn is not None:
        model.optimize(callback_fn)
    else:
        model.optimize()
    end_wall_time = time.perf_counter()

    status_code = model.Status
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    status_str = status_map.get(status_code, f"UNKNOWN_{status_code}")

    has_solution = model.SolCount > 0

    objective: Optional[float] = None
    best_bound: Optional[float] = None
    gap: Optional[float] = None
    partitions_dict: Dict[int, List[Any]] = {p: [] for p in range(1, k + 1)}
    assignment_dict: Dict[Any, int] = {}
    cut_edges_list: List[Tuple[Any, Any]] = []
    is_connected = False
    balance_sizes: Dict[int, int] = {p: 0 for p in range(1, k + 1)}

    if has_solution:
        objective = round(model.ObjVal)
        best_bound = model.ObjBound
        gap = model.MIPGap

        idx_to_node = meta["idx_to_node"]
        x = meta["x"]
        y = meta["y"]

        # Extraer asignación de nodos
        for i in meta["internal_nodes"]:
            orig_node = idx_to_node[i]
            assigned_p = None
            for p in range(1, k + 1):
                if x[i, p].X > 0.5:
                    assigned_p = p
                    break
            if assigned_p is not None:
                partitions_dict[assigned_p].append(orig_node)
                assignment_dict[orig_node] = assigned_p

        # Extraer aristas cortadas
        for i, j in meta["internal_edges"]:
            if y[i, j].X > 0.5:
                u_orig, v_orig = idx_to_node[i], idx_to_node[j]
                cut_edges_list.append((u_orig, v_orig))

        # Verificar conexidad de cada partición en el grafo original
        connected_checks = []
        for p, members in partitions_dict.items():
            balance_sizes[p] = len(members)
            if len(members) == 0:
                connected_checks.append(False)
            else:
                sub = graph.subgraph(members)
                connected_checks.append(nx.is_connected(sub))

        is_connected = all(connected_checks)
    else:
        try:
            best_bound = model.ObjBound
        except (AttributeError, gp.GurobiError):
            best_bound = None

    return {
        "status": status_str,
        "status_code": status_code,
        "has_solution": has_solution,
        "objective": objective,
        "best_bound": best_bound,
        "mip_gap": gap,
        "runtime": model.Runtime if hasattr(model, "Runtime") else (end_wall_time - start_wall_time),
        "wall_time": end_wall_time - start_wall_time,
        "node_count": int(model.NodeCount) if hasattr(model, "NodeCount") else 0,
        "lazy_cuts_added": meta["lazy_cuts_count"],
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
        "model": model,
    }
