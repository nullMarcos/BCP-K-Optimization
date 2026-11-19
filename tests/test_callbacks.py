"""
Pruebas unitarias para separadores nodales y callbacks de conectividad (src/callbacks.py).

Verifica:
    - find_minimal_separator: Poda exacta de separadores candidatos a minimales.
    - bcp_connectivity_callback: Generación y efectividad de lazy cuts en grafos con
      componentes desconectadas potenciales (ej. malla 2D).
"""

import networkx as nx

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
