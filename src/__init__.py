"""
BCP-k Optimization Package.

Particionamiento Balanceado y Conexo de Grafos en k Grupos (BCP_k).

Utiliza importación perezosa (lazy imports) para evitar cargar gurobipy o docplex
a menos que se invoquen explícitamente sus funciones, garantizando desacoplamiento
total entre solvers y facilitando la ejecución en entornos con solo uno de ellos.
"""

from typing import Any

__all__ = [
    "configurar_licencia",
    "find_minimal_separator",
    "find_connectivity_cuts",
    "bcp_connectivity_callback",
    "compute_balance_bounds",
    "prepare_bcp_graph_data",
    "build_solution_summary",
    "build_bcp_gurobi_model",
    "solve_bcp_gurobi",
    "build_bcp_cplex_model",
    "solve_bcp_cplex",
    "solve_bcp",
]


def __getattr__(name: str) -> Any:
    if name in ("find_minimal_separator", "find_connectivity_cuts", "bcp_connectivity_callback"):
        import src.callbacks as _callbacks
        return getattr(_callbacks, name)
    if name in (
        "compute_balance_bounds",
        "prepare_bcp_graph_data",
        "build_solution_summary",
        "solve_bcp",
    ):
        import src.utils as _utils
        return getattr(_utils, name)
    if name in (
        "solve_bcp_gurobi",
        "build_bcp_gurobi_model",
    ):
        import src.model_gurobi as _model_gurobi
        return getattr(_model_gurobi, name)
    if name in (
        "solve_bcp_cplex",
        "build_bcp_cplex_model",
    ):
        import src.model_cplex as _model_cplex
        return getattr(_model_cplex, name)
    if name == "configurar_licencia":
        import src.license as _license
        return getattr(_license, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
