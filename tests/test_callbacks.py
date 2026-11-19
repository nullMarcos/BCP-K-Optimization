"""
Pruebas unitarias para separadores nodales y callbacks de conectividad (src/callbacks.py).

Verifica:
    - find_minimal_separator: Poda exacta de separadores candidatos a minimales.
    - bcp_connectivity_callback: Generación y efectividad de lazy cuts en grafos con
      componentes desconectadas potenciales (ej. malla 2D).
"""

import networkx as nx
import pytest

from src.callbacks import find_minimal_separator
from src.model_gurobi import solve_bcp_gurobi


def test_find_minimal_separator_pruning():
    """
    Verifica que find_minimal_separator pode nodos redundantes de un separador candidato.
    Grafo: Un camino 1 - 2 - 3 - 4 - 5.
    Para desconectar 1 de 5, el conjunto {2, 3, 4} es un separador candidato redundante.
    El algoritmo debe podarlo a un único nodo (separador minimal).
    """
    g = nx.path_graph([1, 2, 3, 4, 5])
    candidate = {2, 3, 4}

    min_sep = find_minimal_separator(g, u=1, v=5, candidate_separator=candidate)

    # Debe ser un subconjunto minimal que aún desconecte 1 y 5
    assert len(min_sep) == 1
    assert min_sep.issubset(candidate)

    # Verificar que remover min_sep efectivamente desconecta 1 y 5
    g_cut = g.copy()
    g_cut.remove_nodes_from(min_sep)
    assert not nx.has_path(g_cut, 1, 5)


def test_find_minimal_separator_already_minimal():
    """Verifica el comportamiento cuando el separador candidato ya es minimal."""
    # Grafo estrella: centro 0 conectado a 1, 2, 3
    g = nx.star_graph(3)
    candidate = {0}

    min_sep = find_minimal_separator(g, u=1, v=2, candidate_separator=candidate)
    assert min_sep == {0}


def test_solve_grid_graph_triggers_lazy_cuts():
    """
    Malla 3x3 con k=3: Requiere agregar lazy cuts para evitar componentes desconectadas.
    Verifica la integración del callback bcp_connectivity_callback en la resolución.
    """
    grid = nx.grid_2d_graph(3, 3)
    res = solve_bcp_gurobi(grid, k=3, epsilon=0.0, verbose=False)

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == 6
    assert res["is_connected"] is True
    assert res["lazy_cuts_added"] > 0

    sizes = sorted(res["balance_stats"]["sizes"].values())
    assert sizes == [3, 3, 3]


def test_callbacks_solver_independence(monkeypatch):
    """
    Verifica que find_minimal_separator sea independiente de gurobipy y que
    bcp_connectivity_callback lance un mensaje informativo si gurobipy no está disponible.
    """
    import importlib
    import src.callbacks as cb

    # Simular un entorno donde gurobipy no está instalado
    monkeypatch.setattr(cb, "_HAS_GUROBI", False)
    monkeypatch.setattr(cb, "gp", None)
    monkeypatch.setattr(cb, "GRB", None)

    # find_minimal_separator debe seguir funcionando con normalidad
    g = nx.path_graph([1, 2, 3])
    min_sep = cb.find_minimal_separator(g, u=1, v=3, candidate_separator={2})
    assert min_sep == {2}

    # bcp_connectivity_callback debe avisar explícitamente sobre la falta de gurobipy
    with pytest.raises(ImportError, match="gurobipy no está disponible"):
        cb.bcp_connectivity_callback(model=None, where=0)
