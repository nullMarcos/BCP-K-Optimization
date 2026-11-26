"""
Módulo del Modelo de Optimización BCP_k en CPLEX (docplex / Branch-and-Cut).

Implementa la formulación de Programación Lineal Entera (PLE) para resolver
el problema de Particionamiento Balanceado y Conexo de Grafos en k Grupos (BCP_k):
    - Variables de decisión binarias x_{i,p} (asignación) e y_{ij} (aristas cortadas).
    - Función objetivo: min sum_{(i,j) in E} y_{ij}.
    - R1: Asignación única por nodo.
    - R2: Balance de tamaño paramétrico ((1 + eps)-balance) con cotas L_min y U_max.
    - R3: Definición lineal de aristas cortadas.
    - R4: Conectividad inducida vía Lazy Constraints (Callback de CPLEX).
    - R5: Dominio binario para variables.
    - R6: Rompimiento de simetrías (Symmetry Breaking) por precedencia dinámica y anclaje.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from docplex.mp.model import Model as DocplexModel

try:
    from cplex import SparsePair
    from cplex.callbacks import Context
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "El paquete 'cplex' es requerido junto con 'docplex' para ejecutar el "
        "modelo BCP_k y registrar el Generic Callback de conectividad."
    ) from exc


# Tipo de función separadora externa para cortes de conectividad R4.
# Recibe (internal_graph, assignment_by_partition) y retorna lista de cortes
# (p, u, v, separator_nodes) donde x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}.
SeparatorFn = Callable[
    [nx.Graph, Dict[int, List[int]]],
    List[Tuple[int, int, int, Set[int]]],
]


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
        raise ValueError(
            f"El número de particiones k debe ser al menos 2, recibido: {k}"
        )
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


def _load_external_separator() -> Optional[SeparatorFn]:
    """
    Intenta cargar dinámicamente la función de separación de conectividad desde
    src.callbacks si se encuentra disponible.
    """
    try:
        from src import callbacks as cb_module  # type: ignore
    except ImportError:
        return None

    for fn_name in (
        "find_connectivity_cuts",
        "separate_connectivity_cuts",
        "get_violated_connectivity_cuts",
    ):
        fn = getattr(cb_module, fn_name, None)
        if callable(fn):
            return fn
    return None


def _default_connectivity_separator(
    internal_graph: nx.Graph,
    partitions_nodes: Dict[int, List[int]],
) -> List[Tuple[int, int, int, Set[int]]]:
    """
    Separador de conectividad por defecto para R4. Para cada partición p con
    múltiples componentes conexas en G[V_p], utiliza la vecindad abierta N(C)
    de cada componente aislada como separador nodal que desconecta un
    representante u in C de un representante v in V_p \\ C.
    """
    cuts: List[Tuple[int, int, int, Set[int]]] = []
    for p, v_p in partitions_nodes.items():
        if len(v_p) <= 1:
            continue

        sub_g = internal_graph.subgraph(v_p)
        components = list(nx.connected_components(sub_g))
        if len(components) <= 1:
            continue

        components.sort(key=len)
        largest_comp = components[-1]
        v_rep_main = next(iter(largest_comp))

        for idx, c_comp in enumerate(components[:-1]):
            u_rep = next(iter(c_comp))
            v_rep = v_rep_main
            # Vecindad abierta de c_comp en el grafo completo (ninguno está en V_p)
            separator: Set[int] = set()
            for node in c_comp:
                for nbr in internal_graph.neighbors(node):
                    if nbr not in c_comp:
                        separator.add(nbr)
            if separator:
                cuts.append((p, u_rep, v_rep, separator))

            # Si hay más de 2 componentes, también separamos la mayor si es útil
            if idx == 0 and len(components) == 2:
                continue

    return cuts


class BCPConnectivityGenericCallback:
    """
    Generic Callback de CPLEX (Context.id.candidate) para restricciones perezosas (R4).

    Compatible con búsqueda dinámica y ejecución paralela multi-hilo en CPLEX.
    En cada solución entera candidata, verifica la conexidad inducida de cada
    subgrafo G[V_p] y rechaza la candidata inyectando las desigualdades de separador
    nodal violadas:
        x_{u,p} + x_{v,p} - 1 <= sum_{s in S} x_{s,p}
    """

    def __init__(
        self,
        bcp_data: Dict[str, Any],
        separator_fn: SeparatorFn = _default_connectivity_separator,
    ) -> None:
        self.bcp_data = bcp_data
        self.separator_fn = separator_fn
        self._lock = threading.Lock()

    def invoke(self, context: Any) -> None:
        if not context.in_candidate() or not context.is_candidate_point():
            return

        inicio = time.perf_counter()
        cuts_added = 0
        try:
            cuts_added = self._separar(context)
        finally:
            transcurrido = time.perf_counter() - inicio
            with self._lock:
                self.bcp_data["lazy_cuts_count"] += cuts_added
                self.bcp_data["callback_time"] = (
                    self.bcp_data.get("callback_time", 0.0) + transcurrido
                )

    def _separar(self, context: Any) -> int:
        data = self.bcp_data
        internal_graph: nx.Graph = data["internal_graph"]
        k: int = data["k"]
        x_indices: Dict[Tuple[int, int], int] = data["x_indices"]
        ordered_keys: List[Tuple[int, int]] = data["x_ordered_keys"]
        ordered_idx: List[int] = data["x_ordered_indices"]

        all_values = context.get_candidate_point()

        partitions_nodes: Dict[int, List[int]] = {p: [] for p in range(1, k + 1)}
        for (i, p), col_idx in zip(ordered_keys, ordered_idx):
            if all_values[col_idx] > 0.5:
                partitions_nodes[p].append(i)

        violated_cuts = self.separator_fn(internal_graph, partitions_nodes)
        if not violated_cuts:
            return 0

        constraints: List[SparsePair] = []
        senses: List[str] = []
        rhs: List[float] = []

        for p, u_rep, v_rep, separator in violated_cuts:
            # Restricción R4: x_{u,p} + x_{v,p} - sum_{s in S} x_{s,p} <= 1
            cut_vars = [x_indices[(u_rep, p)], x_indices[(v_rep, p)]]
            cut_coefs = [1.0, 1.0]
            for s in separator:
                cut_vars.append(x_indices[(s, p)])
                cut_coefs.append(-1.0)

            constraints.append(SparsePair(ind=cut_vars, val=cut_coefs))
            senses.append("L")
            rhs.append(1.0)

        context.reject_candidate(
            constraints=constraints,
            senses=senses,
            rhs=rhs,
        )
        return len(constraints)


# Alias para compatibilidad hacia atrás
BCPConnectivityLazyCallback = BCPConnectivityGenericCallback


def build_bcp_cplex_model(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    model_name: str = "BCP_k_CPLEX",
) -> Tuple[DocplexModel, Dict[str, Any]]:
    """
    Construye el modelo de Programación Lineal Entera (PLE) en docplex/CPLEX.

    Incluye las siguientes restricciones:
      - (R1) Asignación única por nodo.
      - (R2) Balance paramétrico (1 + eps)-balance con cotas enteras L_min y U_max.
      - (R3) Definición lineal de aristas cortadas y_{ij}.
      - (R4) Conectividad inducida (habilitada mediante LazyConstraintCallback).
      - (R5) Naturaleza binaria de x_{i,p} e y_{ij}.
      - (R6) Rompimiento de simetrías por ordenamiento canónico y anclaje de prefijo.

    Args:
        graph: Grafo no dirigido y conexo (NetworkX).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa de balance en [0, 1).
        use_symmetry_breaking: Si es True, agrega las restricciones R6.
        use_lazy_connectivity: Si es True, marca el modelo para usar R4 vía callback.
        model_name: Nombre interno asignado al modelo de docplex.

    Returns:
        Tuple[DocplexModel, Dict[str, Any]]:
            - model: Instancia de docplex.mp.model.Model.
            - metadata: Diccionario con mappings de nodos, variables, cotas y parámetros.
    """
    if graph.is_directed():
        raise ValueError("El problema BCP_k requiere un grafo no dirigido.")
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise ValueError("El grafo de entrada debe ser no vacío y conexo.")

    orig_nodes = list(graph.nodes())
    n = len(orig_nodes)

    if k < 2:
        raise ValueError(
            f"El número de particiones k ({k}) debe ser mayor o igual a 2."
        )
    if n < k:
        raise ValueError(f"El número de nodos ({n}) debe ser mayor o igual a k ({k}).")

    # Mapeo bidireccional nodos originales <-> índices canónicos 1..n
    node_to_idx = {node: idx + 1 for idx, node in enumerate(orig_nodes)}
    idx_to_node = {idx + 1: node for idx, node in enumerate(orig_nodes)}

    # Grafo interno con nodos 1..n y aristas normalizadas (i < j), sin lazos
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

    # Cotas analíticas de balance (R2)
    l_min, u_max = compute_balance_bounds(n, k, epsilon)
    partitions = list(range(1, k + 1))

    # Inicializar modelo docplex
    model = DocplexModel(name=model_name)
    
    # 1. Variables de decisión (R5)
    # x[i, p] in {0, 1}: nodo i asignado a la partición p
    x = model.binary_var_dict(
        [(i, p) for i in internal_nodes for p in partitions],
        name=lambda key: f"x_{key[0]}_{key[1]}",
    )

    # y[i, j] in {0, 1}: arista (i, j) cortada
    y = model.binary_var_dict(
        internal_edges,
        name=lambda key: f"y_{key[0]}_{key[1]}",
    )

    # 2. Función Objetivo: min sum_{(i,j) in E} y_{ij}
    model.minimize(model.sum(y[i, j] for (i, j) in internal_edges))

    # 3. Restricciones
    # (R1) Asignación única: sum_{p=1}^k x_{i,p} = 1  para todo i in V
    for i in internal_nodes:
        model.add_constraint(
            model.sum(x[i, p] for p in partitions) == 1,
            ctname=f"R1_assign_{i}",
        )

    # (R2) Balance de tamaño paramétrico ((1 + eps)-balance)
    for p in partitions:
        part_size = model.sum(x[i, p] for i in internal_nodes)
        model.add_constraint(part_size >= l_min, ctname=f"R2_lower_{p}")
        model.add_constraint(part_size <= u_max, ctname=f"R2_upper_{p}")

    # (R3) Definición lineal de aristas cortadas
    for i, j in internal_edges:
        for p in partitions:
            model.add_constraint(
                y[i, j] >= x[i, p] - x[j, p],
                ctname=f"R3_cut_pos_{i}_{j}_{p}",
            )
            model.add_constraint(
                y[i, j] >= x[j, p] - x[i, p],
                ctname=f"R3_cut_neg_{i}_{j}_{p}",
            )

    # (R6) Rompimiento de simetrías (Symmetry Breaking)
    if use_symmetry_breaking:
        # Anclaje canónico del nodo 1 en la partición 1
        model.add_constraint(x[1, 1] == 1, ctname="R6_anchor_first")

        # Anclaje de prefijo: x_{i,p} = 0 para todo i < p, p in {2..k}
        for p in range(2, k + 1):
            for i in range(1, p):
                model.add_constraint(x[i, p] == 0, ctname=f"R6_prefix_{i}_{p}")

        # Precedencia dinámica: sum_{j < i} x_{j, p-1} >= x_{i, p}
        for i in range(2, n + 1):
            for p in range(2, k + 1):
                model.add_constraint(
                    model.sum(x[j, p - 1] for j in range(1, i)) >= x[i, p],
                    ctname=f"R6_precedence_{i}_{p}",
                )

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
        "callback_time": 0.0,
    }

    return model, metadata


def _normalize_cplex_status(model: DocplexModel, has_solution: bool) -> Tuple[str, int]:
    """
    Normaliza los códigos de estado de CPLEX al mismo vocabulario estándar que Gurobi
    (OPTIMAL, TIME_LIMIT, INFEASIBLE, UNBOUNDED, NODE_LIMIT, INTERRUPTED, etc.)
    para facilitar la comparación directa y la validación cruzada.
    """
    details = model.solve_details
    if details is None:
        return "UNKNOWN", -1

    code = int(details.status_code)
    raw_status = (details.status or "").lower()

    # Códigos estándar de MIP en CPLEX (CPXMIP_*)
    # 101: optimal, 102: optimal_tol, 103: infeasible, 104: sol_lim,
    # 105/106: node_lim_feas/infeas, 107/108: time_lim_feas/infeas,
    # 111/112: mem_lim, 113/114: abort_user, 118: unbounded, 119: inf_or_unbd
    if code in (1, 101, 102) or "integer optimal" in raw_status:
        return "OPTIMAL", code
    if code in (107, 108, 11, 25) or "time limit" in raw_status:
        return "TIME_LIMIT", code
    if code in (3, 103) or "infeasible" in raw_status and "unbounded" not in raw_status:
        return "INFEASIBLE", code
    if code in (4, 119) or "infeasible or unbounded" in raw_status:
        return "INF_OR_UNBD", code
    if code in (2, 118) or "unbounded" in raw_status:
        return "UNBOUNDED", code
    if code in (105, 106) or "node limit" in raw_status:
        return "NODE_LIMIT", code
    if code in (113, 114, 13) or "abort" in raw_status:
        return "INTERRUPTED", code

    if has_solution:
        return f"FEASIBLE_{code}", code
    return f"UNKNOWN_{code}", code


def solve_bcp_cplex(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    presolve: Optional[int] = None,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    log_file: Optional[str] = None,
    verbose: bool = True,
    separator_fn: Optional[SeparatorFn] = None,
    keep_model: bool = False,
) -> Dict[str, Any]:
    """
    Construye y resuelve el modelo BCP_k en CPLEX (docplex) con soporte de Lazy
    Constraints para conectividad inducida (R4).

    Args:
        graph: Grafo no dirigido y conexo (NetworkX).
        k: Número de particiones (k >= 2).
        epsilon: Tolerancia relativa de balance en [0, 1).
        time_limit: Límite de tiempo en segundos (timelimit de CPLEX).
        mip_gap: Tolerancia relativa de brecha de optimalidad (mip.tolerances.mipgap).
        threads: Número máximo de hilos paralelos (threads de CPLEX).
        presolve: Nivel de presolve de CPLEX (0=desactivado, 1=activado por defecto).
                  Si es None, mantiene la configuración predeterminada del solver.
        use_symmetry_breaking: Si es True, incluye restricciones R6.
        use_lazy_connectivity: Si es True, registra el callback de conectividad R4.
        log_file: Ruta opcional para guardar el log de CPLEX (ej. outputs/logs/...).
        verbose: Si es False, silencia la salida estándar de CPLEX.
        separator_fn: Función separadora personalizada opcional. Si es None, busca
                      automáticamente en src/callbacks.py y utiliza el separador
                      por defecto si no está definido.
        keep_model: Si es True, conserva y retorna la instancia `model` sin cerrarla;
                    de lo contrario libera recursos de CPLEX al finalizar.

    Returns:
        Dict[str, Any]: Diccionario homogéneo con Gurobi con métricas y solución:
            - solver: "CPLEX"
            - status: Estado normalizado (OPTIMAL, TIME_LIMIT, INFEASIBLE, etc.).
            - status_code: Código numérico nativo de CPLEX.
            - has_solution: Booleano indicando si se encontró al menos una solución factible.
            - objective: Valor objetivo entero (aristas cortadas) o None.
            - best_bound: Mejor cota dual (best_bound) reportada por CPLEX.
            - mip_gap: Brecha de optimalidad relativa final (en [0, 1], ej. 0.05 = 5%).
            - runtime: Tiempo de cómputo reportado por CPLEX en segundos.
            - wall_time: Tiempo de pared medido externamente en segundos.
            - node_count: Número de nodos procesados en el árbol Branch-and-Cut.
            - lazy_cuts_added: Cantidad de restricciones perezosas R4 inyectadas.
            - callback_time: Segundos usados dentro del callback de conectividad.
            - partitions: Mapeo {p: [nodos]} con etiquetas originales del grafo.
            - assignment: Mapeo {nodo: p} para cada nodo del grafo.
            - cut_edges: Lista de tuplas (u, v) de aristas cortadas.
            - is_connected: Booleano que certifica si cada partición es conexa.
            - balance_stats: Dict con tamaños por bloque {p: tamaño}, l_min, u_max y epsilon.
            - model: Instancia de DocplexModel si keep_model=True, si no None.
    """
    model, meta = build_bcp_cplex_model(
        graph=graph,
        k=k,
        epsilon=epsilon,
        use_symmetry_breaking=use_symmetry_breaking,
        use_lazy_connectivity=use_lazy_connectivity,
    )

    # Configuración de parámetros de CPLEX
    if time_limit is not None:
        model.parameters.timelimit = max(0.0, float(time_limit))
    if mip_gap is not None:
        model.parameters.mip.tolerances.mipgap = max(0.0, float(mip_gap))
    if threads is not None:
        model.parameters.threads = max(1, int(threads))
    if presolve is not None:
        model.parameters.preprocessing.presolve = int(presolve)

    # Registro de Generic Callback (Context.id.candidate) para R4 con soporte multi-hilo
    if use_lazy_connectivity:
        x = meta["x"]
        ordered_keys = list(x.keys())
        x_indices = {key: var.index for key, var in x.items()}
        meta["x_indices"] = x_indices
        meta["x_ordered_keys"] = ordered_keys
        meta["x_ordered_indices"] = [x_indices[key] for key in ordered_keys]

        active_separator = (
            separator_fn
            or _load_external_separator()
            or _default_connectivity_separator
        )
        cb = BCPConnectivityGenericCallback(
            bcp_data=meta,
            separator_fn=active_separator,
        )
        cpx = model.get_cplex()
        cpx.set_callback(cb, Context.id.candidate)

    # Manejo de log de salida (consola y/o archivo en outputs/logs/)
    log_stream = None
    log_output: Any = verbose
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8")
        log_output = log_stream

    start_wall_time = time.perf_counter()
    try:
        solution = model.solve(log_output=log_output)
    finally:
        if log_stream is not None:
            log_stream.close()
    end_wall_time = time.perf_counter()

    has_solution = solution is not None
    status_str, status_code = _normalize_cplex_status(model, has_solution)
    details = model.solve_details

    objective: Optional[int] = None
    best_bound: Optional[float] = None
    gap: Optional[float] = None
    runtime: float = (
        float(details.time)
        if details is not None and details.time is not None
        else (end_wall_time - start_wall_time)
    )
    node_count: int = (
        int(details.nb_nodes_processed)
        if details is not None and details.nb_nodes_processed is not None
        else 0
    )

    partitions_dict: Dict[int, List[Any]] = {p: [] for p in range(1, k + 1)}
    assignment_dict: Dict[Any, int] = {}
    cut_edges_list: List[Tuple[Any, Any]] = []
    is_connected = False
    balance_sizes: Dict[int, int] = {p: 0 for p in range(1, k + 1)}

    if has_solution:
        objective = int(round(solution.objective_value))
        if details is not None:
            best_bound = (
                float(details.best_bound) if details.best_bound is not None else None
            )
            gap = (
                float(details.mip_relative_gap)
                if details.mip_relative_gap is not None
                else None
            )

        idx_to_node = meta["idx_to_node"]
        x = meta["x"]
        y = meta["y"]

        # Extraer asignación de nodos
        for i in meta["internal_nodes"]:
            orig_node = idx_to_node[i]
            assigned_p = None
            for p in range(1, k + 1):
                if solution.get_value(x[i, p]) > 0.5:
                    assigned_p = p
                    break
            if assigned_p is not None:
                partitions_dict[assigned_p].append(orig_node)
                assignment_dict[orig_node] = assigned_p

        # Extraer aristas cortadas
        for i, j in meta["internal_edges"]:
            if solution.get_value(y[i, j]) > 0.5:
                u_orig, v_orig = idx_to_node[i], idx_to_node[j]
                cut_edges_list.append((u_orig, v_orig))

        # Verificar conexidad inducida de cada partición en el grafo original
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
        if details is not None and details.best_bound is not None:
            best_bound = float(details.best_bound)

    result: Dict[str, Any] = {
        "solver": "CPLEX",
        "status": status_str,
        "status_code": status_code,
        "has_solution": has_solution,
        "objective": objective,
        "best_bound": best_bound,
        "mip_gap": gap,
        "runtime": runtime,
        "wall_time": end_wall_time - start_wall_time,
        "node_count": node_count,
        "lazy_cuts_added": meta["lazy_cuts_count"],
        "callback_time": meta["callback_time"],
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
        "model": model if keep_model else None,
    }

    if not keep_model:
        model.end()

    return result


def solve_bcp(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    time_limit: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Interfaz estándar de resolución BCP_k para el solver CPLEX.
    """
    return solve_bcp_cplex(
        graph=graph,
        k=k,
        epsilon=epsilon,
        time_limit=time_limit,
        **kwargs,
    )
