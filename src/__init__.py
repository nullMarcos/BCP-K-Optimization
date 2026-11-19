"""
BCP-k Optimization Package.

Particionamiento Balanceado y Conexo de Grafos en k Grupos (BCP_k).
"""

from src.license import configurar_licencia
from src.callbacks import find_minimal_separator, bcp_connectivity_callback
from src.model_gurobi import (
    compute_balance_bounds,
    build_bcp_gurobi_model,
    solve_bcp_gurobi,
    solve_bcp,
)

__all__ = [
    "configurar_licencia",
    "find_minimal_separator",
    "bcp_connectivity_callback",
    "compute_balance_bounds",
    "build_bcp_gurobi_model",
    "solve_bcp_gurobi",
    "solve_bcp",
]
