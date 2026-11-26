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

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from src.callbacks import bcp_connectivity_callback, find_minimal_separator
from src.utils import (
    build_solution_summary,
    compute_balance_bounds,
    configurar_licencia,
    prepare_bcp_graph_data,
    solve_bcp as _unified_solve_bcp,
)


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
    graph_data = prepare_bcp_graph_data(graph, k=k, epsilon=epsilon)
    n = graph_data["n"]
    l_min = graph_data["l_min"]
    u_max = graph_data["u_max"]
    partitions = graph_data["partitions"]
    internal_nodes = graph_data["internal_nodes"]
    internal_edges = graph_data["internal_edges"]

    # Inicializar modelo Gurobi con el entorno especificado o el entorno por defecto
    model = gp.Model(model_name, env=env) if env is not None else gp.Model(model_name)

    # 1. Variables de Decisión (R5)
    x = model.addVars(
        internal_nodes,
        partitions,
        vtype=GRB.BINARY,
        name="x",
    )
    y = model.addVars(
        internal_edges,
        vtype=GRB.BINARY,
        name="y",
    )

    # 2. Función Objetivo: Minimizar Z = sum_{(i,j) in E} y_{ij}
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
        model.addConstr(x[1, 1] == 1, name="R6_anchor_first")

        for p in range(2, k + 1):
            for i in range(1, p):
                model.addConstr(x[i, p] == 0, name=f"R6_prefix_{i}_{p}")

        for i in range(2, n + 1):
            for p in range(2, k + 1):
                model.addConstr(
                    gp.quicksum(x[j, p - 1] for j in range(1, i)) >= x[i, p],
                    name=f"R6_precedence_{i}_{p}",
                )

    if use_lazy_connectivity:
        model.Params.LazyConstraints = 1

    metadata: Dict[str, Any] = {
        **graph_data,
        "x": x,
        "y": y,
        "use_symmetry_breaking": use_symmetry_breaking,
        "use_lazy_connectivity": use_lazy_connectivity,
        "lazy_cuts_count": 0,
        "callback_time": 0.0,
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
    presolve: Optional[int] = None,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    log_file: Optional[str] = None,
    verbose: bool = True,
    env: Optional[gp.Env] = None,
    lic_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construye y resuelve el modelo BCP_k en Gurobi con callbacks perezosos.
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

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogToConsole = 1 if verbose else 0
        model.Params.OutputFlag = 1
        model.Params.LogFile = str(log_file)
    else:
        model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = max(0.0, float(time_limit))
    if mip_gap is not None:
        model.Params.MIPGap = max(0.0, float(mip_gap))
    if threads is not None:
        model.Params.Threads = max(1, int(threads))
    if presolve is not None:
        model.Params.Presolve = int(presolve)

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

    if has_solution:
        objective = round(model.ObjVal)
        best_bound = model.ObjBound
        gap = model.MIPGap
        x = meta["x"]
        y = meta["y"]
        sol_summary = build_solution_summary(
            graph=graph,
            meta=meta,
            is_x_active=lambda i, p: x[i, p].X > 0.5,
            is_y_active=lambda i, j: y[i, j].X > 0.5,
        )
    else:
        try:
            best_bound = model.ObjBound
        except (AttributeError, gp.GurobiError):
            best_bound = None
        sol_summary = build_solution_summary(graph=graph, meta=meta)

    return {
        "solver": "GUROBI",
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
        "callback_time": meta["callback_time"],
        **sol_summary,
        "model": model,
    }


def solve_bcp(
    graph: nx.Graph,
    k: int,
    epsilon: float = 0.0,
    time_limit: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Delega en el despachador unificado `src.utils.solve_bcp` usando Gurobi por defecto."""
    return _unified_solve_bcp(
        graph=graph,
        k=k,
        epsilon=epsilon,
        time_limit=time_limit,
        solver="gurobi",
        **kwargs,
    )
