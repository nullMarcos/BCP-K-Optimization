"""
Pruebas unitarias para el modelo de optimización BCP_k en Gurobi (src/model_gurobi.py).
Utiliza el módulo estándar unittest para compatibilidad universal sin dependencias externas.
"""

import networkx as nx
import pytest

from src.model_gurobi import (
    compute_balance_bounds,
    configurar_licencia,
    solve_bcp_gurobi,
    solve_bcp,
)

def test_configurar_licencia(tmp_path):
    """Verifica que configurar_licencia valide rutas y lance excepciones apropiadas."""
    # Ruta inexistente -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        configurar_licencia(tmp_path / "no_existe" / "gurobi.lic", verbose=False)

    # Archivo vacío -> ValueError
    licencia_vacia = tmp_path / "vacia.lic"
    licencia_vacia.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        configurar_licencia(licencia_vacia, verbose=False)

    # Archivo sin clave KEY -> ValueError
    licencia_sin_key = tmp_path / "sin_key.lic"
    licencia_sin_key.write_text("HOSTNAME=test\nLICENSEID=12345\n", encoding="utf-8")
    with pytest.raises(ValueError):
        configurar_licencia(licencia_sin_key, verbose=False)


def test_compute_balance_bounds_paper_examples():
    """Valida el cálculo analítico de cotas contra los ejemplos del documento PDF."""
    # Ejemplo IEEE 14 barras (n=14, k=3)
    # Balance estricto eps=0.0 -> L_min = 4, U_max = 5
    l_min, u_max = compute_balance_bounds(14, 3, epsilon=0.0)
    assert l_min == 4
    assert u_max == 5

    # Con tolerancia eps=0.15 (15%) -> L_min = 3, U_max = 6
    l_min_eps, u_max_eps = compute_balance_bounds(14, 3, epsilon=0.15)
    assert l_min_eps == 3
    assert u_max_eps == 6


def test_compute_balance_bounds_validation():
    """Verifica que se lancen excepciones ante parámetros inválidos."""
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 0)
    with pytest.raises(ValueError):
        compute_balance_bounds(5, 10)  # n < k
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 2, epsilon=1.0)
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 2, epsilon=-0.1)


@pytest.mark.parametrize(
    ("graph_factory", "k", "expected_obj", "expected_cut_len", "expected_sizes"),
    [
        # Camino P6 con k=2 y balance estricto: corta 1 arista y deja particiones [3, 3]
        (lambda: nx.path_graph(6), 2, 1, 1, [3, 3]),
        # Ciclo C6 con k=2: corta 2 aristas y deja particiones [3, 3]
        (lambda: nx.cycle_graph(6), 2, 2, 2, [3, 3]),
        # Barbell (4, 0): corta el puente central (1 arista) y deja particiones [4, 4]
        (lambda: nx.barbell_graph(4, 0), 2, 1, 1, [4, 4]),
    ],
)
def test_solve_small_graphs(graph_factory, k, expected_obj, expected_cut_len, expected_sizes):
    """Verifica cortes óptimos, conectividad y balance en grafos canónicos."""
    graph = graph_factory()
    res = solve_bcp_gurobi(graph, k=k, epsilon=0.0, verbose=False)

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == expected_obj
    assert res["is_connected"] is True
    assert len(res["cut_edges"]) == expected_cut_len

    sizes = sorted(res["balance_stats"]["sizes"].values())
    assert sizes == expected_sizes


def test_symmetry_breaking_anchor():
    """Verifica que el rompimiento de simetrías asigne de manera determinista al primer nodo."""
    p4 = nx.path_graph(4)
    res = solve_bcp_gurobi(p4, k=2, epsilon=0.0, use_symmetry_breaking=True, verbose=False)

    assert res["status"] == "OPTIMAL"
    first_node = next(iter(p4.nodes()))
    assert res["assignment"][first_node] == 1


def test_solve_bcp_unified_wrapper():
    """Verifica que la función unificada solve_bcp opere de forma idéntica a solve_bcp_gurobi."""
    p6 = nx.path_graph(6)
    res = solve_bcp(p6, k=2, epsilon=0.0, time_limit=10.0, verbose=False)

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == 1
    assert res["is_connected"] is True
    assert len(res["cut_edges"]) == 1


def test_solve_bcp_gurobi_presolve():
    """Verifica que el parámetro presolve se configure y ejecute correctamente en Gurobi."""
    p6 = nx.path_graph(6)
    # Probar con Presolve desactivado (0) y agresivo (2)
    res_no_presolve = solve_bcp_gurobi(p6, k=2, presolve=0, verbose=False)
    assert res_no_presolve["status"] == "OPTIMAL"
    assert res_no_presolve["model"].Params.Presolve == 0

    res_aggressive = solve_bcp_gurobi(p6, k=2, presolve=2, verbose=False)
    assert res_aggressive["status"] == "OPTIMAL"
    assert res_aggressive["model"].Params.Presolve == 2
