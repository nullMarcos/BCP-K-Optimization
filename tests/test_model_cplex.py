"""
Pruebas unitarias para el modelo BCP_k en CPLEX (src/model_cplex.py) y validación
cruzada contra Gurobi cuando src/model_gurobi.py y una licencia válida estén disponibles.

"""

from __future__ import annotations

from pathlib import Path
import networkx as nx
import pytest

from src.model_cplex import (
    build_bcp_cplex_model,
    compute_balance_bounds,
    solve_bcp,
    solve_bcp_cplex,
)


def _gurobi_available() -> bool:
    """Verifica si src/model_gurobi.py y una licencia operativa de Gurobi están disponibles."""
    try:
        import gurobipy as gp  # noqa: F401
        from src.model_gurobi import solve_bcp_gurobi  # noqa: F401
        return True
    except Exception:
        return False


def test_cplex_compute_balance_bounds_examples():
    """Valida el cálculo analítico de cotas R2 para n=14, k=3 con distintos valores de epsilon."""
    # IEEE 14 barras (n=14, k=3)
    # Balance estricto eps=0.0 -> L_min = 4, U_max = 5
    l_min, u_max = compute_balance_bounds(14, 3, epsilon=0.0)
    assert l_min == 4
    assert u_max == 5

    # Tolerancia paramétrica eps=0.15 -> L_min = 3, U_max = 6
    l_min_eps, u_max_eps = compute_balance_bounds(14, 3, epsilon=0.15)
    assert l_min_eps == 3
    assert u_max_eps == 6


def test_cplex_compute_balance_bounds_validation():
    """Verifica manejo de excepciones ante parámetros inválidos en compute_balance_bounds."""
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 1)  # k < 2
    with pytest.raises(ValueError):
        compute_balance_bounds(5, 10)  # n < k
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 2, epsilon=1.0)
    with pytest.raises(ValueError):
        compute_balance_bounds(10, 2, epsilon=-0.1)


def test_cplex_build_model_validations():
    """Verifica que build_bcp_cplex_model rechace grafos dirigidos o desconectados."""
    dg = nx.DiGraph([(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        build_bcp_cplex_model(dg, k=2)

    disconnected = nx.Graph()
    disconnected.add_edges_from([(0, 1), (2, 3)])
    with pytest.raises(ValueError):
        build_bcp_cplex_model(disconnected, k=2)


@pytest.mark.parametrize(
    ("graph_factory", "k", "expected_obj", "expected_cut_len", "expected_sizes"),
    [
        # Camino P6 con k=2 y balance estricto: corta 1 arista y deja bloques [3, 3]
        (lambda: nx.path_graph(6), 2, 1, 1, [3, 3]),
        # Ciclo C6 con k=2: corta 2 aristas y deja bloques [3, 3]
        (lambda: nx.cycle_graph(6), 2, 2, 2, [3, 3]),
        # Barbell (4, 0): corta el puente central (1 arista) y deja bloques [4, 4]
        (lambda: nx.barbell_graph(4, 0), 2, 1, 1, [4, 4]),
        # Malla 3x3 con k=3: óptimo 6 aristas cortadas y bloques conexos [3, 3, 3]
        (lambda: nx.grid_2d_graph(3, 3), 3, 6, 6, [3, 3, 3]),
    ],
)
def test_cplex_solve_canonical_graphs(
    graph_factory, k, expected_obj, expected_cut_len, expected_sizes
):
    """Verifica optimalidad, conectividad y balance exacto en grafos canónicos."""
    graph = graph_factory()
    res = solve_bcp_cplex(graph, k=k, epsilon=0.0, verbose=False)

    assert res["solver"] == "CPLEX"
    assert res["status"] == "OPTIMAL"
    assert res["has_solution"] is True
    assert res["objective"] == expected_obj
    assert res["is_connected"] is True
    assert len(res["cut_edges"]) == expected_cut_len
    assert sorted(res["balance_stats"]["sizes"].values()) == expected_sizes


def test_cplex_symmetry_breaking_canonical_order():
    """Verifica que R6 fije el nodo 1 en la partición 1 y cumpla precedencia canónica."""
    g = nx.grid_2d_graph(3, 3)
    res = solve_bcp_cplex(g, k=3, epsilon=0.0, use_symmetry_breaking=True, verbose=False)

    assert res["status"] == "OPTIMAL"
    nodes = list(g.nodes())
    node_to_idx = {node: idx + 1 for idx, node in enumerate(nodes)}

    # Anclaje inicial: primer nodo en partición 1
    assert res["assignment"][nodes[0]] == 1

    # Los índices mínimos de cada partición deben quedar estrictamente crecientes
    min_indices = [
        min(node_to_idx[node] for node in res["partitions"][p])
        for p in range(1, 4)
    ]
    assert min_indices == sorted(min_indices)
    assert len(set(min_indices)) == 3


def test_cplex_symmetry_breaking_preserves_optimum():
    """Verifica que activar R6 no altera el valor objetivo óptimo respecto al modelo sin R6."""
    g = nx.karate_club_graph()
    res_with_sb = solve_bcp_cplex(
        g, k=2, epsilon=0.1, use_symmetry_breaking=True, time_limit=30, verbose=False
    )
    res_without_sb = solve_bcp_cplex(
        g, k=2, epsilon=0.1, use_symmetry_breaking=False, time_limit=30, verbose=False
    )

    assert res_with_sb["status"] == "OPTIMAL"
    assert res_without_sb["status"] == "OPTIMAL"
    assert res_with_sb["objective"] == res_without_sb["objective"]
    assert res_with_sb["is_connected"] is True
    assert res_without_sb["is_connected"] is True


def test_cplex_epsilon_tradeoff_and_infeasibility():
    """
    Grafo estrella S_6 (1 centro + 5 hojas) con k=2:
    - Con eps=0.0 (tamaños [3, 3]) es infactible porque cualquier bloque conexo sin el centro
      puede tener a lo sumo 1 nodo.
    - Con eps=0.70 (L_min=1, U_max=6) se vuelve factible con 1 arista cortada ([1, 5]).
    """
    star = nx.star_graph(5)  # 6 nodos totales
    res_strict = solve_bcp_cplex(star, k=2, epsilon=0.0, verbose=False)
    assert res_strict["status"] == "INFEASIBLE"
    assert res_strict["has_solution"] is False

    res_relaxed = solve_bcp_cplex(star, k=2, epsilon=0.70, verbose=False)
    assert res_relaxed["status"] == "OPTIMAL"
    assert res_relaxed["objective"] == 1
    assert res_relaxed["is_connected"] is True
    assert sorted(res_relaxed["balance_stats"]["sizes"].values()) == [1, 5]


def test_cplex_parameters_and_log_file(tmp_path: Path):
    """Verifica sintonización de parámetros (presolve, threads, mip_gap, log_file) y firma solve_bcp."""
    g = nx.cycle_graph(8)
    log_path = tmp_path / "cplex_test.log"

    res = solve_bcp(
        graph=g,
        k=2,
        epsilon=0.0,
        time_limit=10.0,
        mip_gap=0.0,
        threads=1,
        presolve=1,
        log_file=str(log_path),
        verbose=False,
    )

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == 2
    assert res["is_connected"] is True
    assert log_path.exists()
    assert log_path.stat().st_size > 0


def test_cplex_multithread_generic_callback(tmp_path: Path):
    """Verifica ejecución paralela multi-hilo con Generic Callback de conectividad."""
    g = nx.karate_club_graph()
    log_path = tmp_path / "cplex_mt.log"

    res = solve_bcp_cplex(
        graph=g,
        k=2,
        epsilon=0.05,
        time_limit=30.0,
        threads=4,
        log_file=str(log_path),
        verbose=False,
    )

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == 10
    assert res["is_connected"] is True
    log_text = log_path.read_text(encoding="utf-8")
    assert "Generic callback" in log_text
    assert "using up to 4 threads" in log_text


@pytest.mark.skipif(
    not _gurobi_available(),
    reason="Validación cruzada omitida: src/model_gurobi.py o gurobipy no disponibles en esta rama.",
)
@pytest.mark.parametrize(
    ("graph_factory", "k", "epsilon"),
    [
        (lambda: nx.path_graph(6), 2, 0.0),
        (lambda: nx.cycle_graph(8), 2, 0.0),
        (lambda: nx.grid_2d_graph(3, 3), 3, 0.0),
        (lambda: nx.karate_club_graph(), 2, 0.1),
    ],
)
def test_cross_validation_cplex_vs_gurobi(graph_factory, k, epsilon):
    """
    Validación cruzada directa: verifica que CPLEX y Gurobi converjan exactamente
    al mismo valor objetivo óptimo Z* y cotas duales consistentes en grafos pequeños.
    """
    from src.model_gurobi import solve_bcp_gurobi

    graph = graph_factory()
    res_cplex = solve_bcp_cplex(
        graph, k=k, epsilon=epsilon, time_limit=60.0, verbose=False
    )
    res_gurobi = solve_bcp_gurobi(
        graph, k=k, epsilon=epsilon, time_limit=60.0, verbose=False
    )

    assert res_cplex["status"] == "OPTIMAL"
    assert res_gurobi["status"] == "OPTIMAL"
    assert res_cplex["objective"] == res_gurobi["objective"]
    assert res_cplex["is_connected"] is True
    assert res_gurobi["is_connected"] is True
    assert abs(res_cplex["best_bound"] - res_gurobi["best_bound"]) <= 1e-4
