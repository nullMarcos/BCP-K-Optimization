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

from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
from docplex.mp.model import Model as DocplexModel

from src.callbacks import find_connectivity_cuts
from src.utils import (
    build_solution_summary,
    compute_balance_bounds,
    prepare_bcp_graph_data,
    solve_bcp as _unified_solve_bcp,
)

try:
    from cplex import SparsePair
    from cplex.callbacks import Context
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "El paquete 'cplex' es requerido junto con 'docplex' para ejecutar el "
        "modelo BCP_k y registrar el Generic Callback de conectividad."
    ) from exc


# Tipo de función separadora externa para cortes de conectividad R4.
SeparatorFn = Callable[
    [nx.Graph, Dict[int, List[int]]],
    List[Tuple[int, int, int, Set[int]]],
]


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
        separator_fn: SeparatorFn = find_connectivity_cuts,
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

        if self.separator_fn is find_connectivity_cuts:
            violated_cuts = find_connectivity_cuts(
                internal_graph, partitions_nodes, deduplicate=True
            )
        else:
            violated_cuts = self.separator_fn(internal_graph, partitions_nodes)
        if not violated_cuts:
            return 0

        constraints: List[SparsePair] = []
        senses: List[str] = []
        rhs: List[float] = []
        seen_cuts: Set[Tuple[int, int, int, frozenset[int]]] = set()

        for p, u_rep, v_rep, separator in violated_cuts:
            pair_u, pair_v = (u_rep, v_rep) if u_rep <= v_rep else (v_rep, u_rep)
            cut_key = (p, pair_u, pair_v, frozenset(separator))
            if cut_key in seen_cuts:
                continue
            seen_cuts.add(cut_key)

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
    """
    graph_data = prepare_bcp_graph_data(graph, k=k, epsilon=epsilon)
    n = graph_data["n"]
    l_min = graph_data["l_min"]
    u_max = graph_data["u_max"]
    partitions = graph_data["partitions"]
    internal_nodes = graph_data["internal_nodes"]
    internal_edges = graph_data["internal_edges"]

    model = DocplexModel(name=model_name)

    # 1. Variables de decisión (R5)
    x = model.binary_var_dict(
        [(i, p) for i in internal_nodes for p in partitions],
        name=lambda key: f"x_{key[0]}_{key[1]}",
    )
    y = model.binary_var_dict(
        internal_edges,
        name=lambda key: f"y_{key[0]}_{key[1]}",
    )

    # 2. Función Objetivo: min sum_{(i,j) in E} y_{ij}
    model.minimize(model.sum(y[i, j] for (i, j) in internal_edges))

    # 3. Restricciones (registradas por lotes mediante add_constraints)
    # (R1) Asignación única
    model.add_constraints(
        (model.sum(x[i, p] for p in partitions) == 1 for i in internal_nodes),
        names=(f"R1_assign_{i}" for i in internal_nodes),
    )

    # (R2) Balance de tamaño paramétrico ((1 + eps)-balance)
    part_sizes = {p: model.sum(x[i, p] for i in internal_nodes) for p in partitions}
    model.add_constraints(
        (part_sizes[p] >= l_min for p in partitions),
        names=(f"R2_lower_{p}" for p in partitions),
    )
    model.add_constraints(
        (part_sizes[p] <= u_max for p in partitions),
        names=(f"R2_upper_{p}" for p in partitions),
    )

    # (R3) Definición lineal de aristas cortadas
    model.add_constraints(
        (y[i, j] >= x[i, p] - x[j, p] for i, j in internal_edges for p in partitions),
        names=(f"R3_cut_pos_{i}_{j}_{p}" for i, j in internal_edges for p in partitions),
    )
    model.add_constraints(
        (y[i, j] >= x[j, p] - x[i, p] for i, j in internal_edges for p in partitions),
        names=(f"R3_cut_neg_{i}_{j}_{p}" for i, j in internal_edges for p in partitions),
    )

    # (R6) Rompimiento de simetrías (Symmetry Breaking)
    if use_symmetry_breaking:
        model.add_constraint(x[1, 1] == 1, ctname="R6_anchor_first")

        model.add_constraints(
            (x[i, p] == 0 for p in range(2, k + 1) for i in range(1, p)),
            names=(f"R6_prefix_{i}_{p}" for p in range(2, k + 1) for i in range(1, p)),
        )

        model.add_constraints(
            (
                model.sum(x[j, p - 1] for j in range(1, i)) >= x[i, p]
                for i in range(2, n + 1)
                for p in range(2, k + 1)
            ),
            names=(
                f"R6_precedence_{i}_{p}"
                for i in range(2, n + 1)
                for p in range(2, k + 1)
            ),
        )

    metadata: Dict[str, Any] = {
        **graph_data,
        "x": x,
        "y": y,
        "use_symmetry_breaking": use_symmetry_breaking,
        "use_lazy_connectivity": use_lazy_connectivity,
        "lazy_cuts_count": 0,
        "callback_time": 0.0,
    }

    return model, metadata


def _normalize_cplex_status(model: DocplexModel, has_solution: bool) -> Tuple[str, int]:
    """
    Normaliza los códigos de estado de CPLEX al mismo vocabulario estándar que Gurobi
    (OPTIMAL, TIME_LIMIT, INFEASIBLE, UNBOUNDED, NODE_LIMIT, INTERRUPTED, etc.).
    """
    details = model.solve_details
    if details is None:
        return "UNKNOWN", -1

    code = int(details.status_code)
    raw_status = (details.status or "").lower()

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
    """
    model, meta = build_bcp_cplex_model(
        graph=graph,
        k=k,
        epsilon=epsilon,
        use_symmetry_breaking=use_symmetry_breaking,
        use_lazy_connectivity=use_lazy_connectivity,
    )

    if time_limit is not None:
        model.parameters.timelimit = max(0.0, float(time_limit))
    if mip_gap is not None:
        model.parameters.mip.tolerances.mipgap = max(0.0, float(mip_gap))
    if threads is not None:
        model.parameters.threads = max(1, int(threads))
    if presolve is not None:
        model.parameters.preprocessing.presolve = int(presolve)

    if use_lazy_connectivity:
        x = meta["x"]
        ordered_keys = list(x.keys())
        x_indices = {key: var.index for key, var in x.items()}
        meta["x_indices"] = x_indices
        meta["x_ordered_keys"] = ordered_keys
        meta["x_ordered_indices"] = [x_indices[key] for key in ordered_keys]

        cb = BCPConnectivityGenericCallback(
            bcp_data=meta,
            separator_fn=separator_fn or find_connectivity_cuts,
        )
        cpx = model.get_cplex()
        cpx.set_callback(cb, Context.id.candidate)

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

        x = meta["x"]
        y = meta["y"]
        sol_summary = build_solution_summary(
            graph=graph,
            meta=meta,
            is_x_active=lambda i, p: solution.get_value(x[i, p]) > 0.5,
            is_y_active=lambda i, j: solution.get_value(y[i, j]) > 0.5,
        )
    else:
        if details is not None and details.best_bound is not None:
            best_bound = float(details.best_bound)
        sol_summary = build_solution_summary(graph=graph, meta=meta)

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
        **sol_summary,
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
    """Delega en el despachador unificado `src.utils.solve_bcp` usando CPLEX."""
    return _unified_solve_bcp(
        graph=graph,
        k=k,
        epsilon=epsilon,
        time_limit=time_limit,
        solver="cplex",
        **kwargs,
    )
