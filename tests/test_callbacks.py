"""
Pruebas unitarias para separadores nodales y callbacks de conectividad (src/callbacks.py).

Verifica:
    - find_minimal_separator: Poda exacta de separadores candidatos a minimales.
    - bcp_connectivity_callback: Generación y efectividad de lazy cuts en grafos con
      componentes desconectadas potenciales (ej. malla 2D).
"""

import itertools
import math
import random

import networkx as nx
import pytest

from src.callbacks import find_connectivity_cuts, find_minimal_separator

try:
    from src.model_gurobi import solve_bcp_gurobi
    HAY_GUROBI = True
except ImportError:
    HAY_GUROBI = False

try:
    from src.model_cplex import solve_bcp_cplex
    HAY_CPLEX = True
except ImportError:
    HAY_CPLEX = False

requiere_gurobi = pytest.mark.skipif(not HAY_GUROBI, reason="gurobipy no instalado")


def _revisar_corte(g, partitions_nodes, corte):
    """Comprueba que un corte (p, u, v, S) sea válido, esté violado y S sea minimal."""
    p, u, v, sep = corte
    v_p = set(partitions_nodes[p])

    assert u in v_p and v in v_p
    # el corte está violado: lado izquierdo = 1 y ningún nodo de S está en V_p
    assert not (sep & v_p)

    # S separa a u de v
    g_sin_s = g.copy()
    g_sin_s.remove_nodes_from(sep)
    assert not nx.has_path(g_sin_s, u, v)

    # minimalidad: si se saca cualquier nodo de S, u y v vuelven a conectarse
    for s in sep:
        g_aux = g.copy()
        g_aux.remove_nodes_from(sep - {s})
        assert nx.has_path(g_aux, u, v)


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


def test_cuts_particion_conexa_no_genera_cortes():
    g = nx.path_graph(range(1, 7))
    particiones = {1: [1, 2, 3], 2: [4, 5, 6]}
    assert find_connectivity_cuts(g, particiones) == []


def test_cuts_particion_desconectada():
    # Camino 1-2-3-4-5-6: la partición 1 queda partida en {1,2} y {5,6}
    g = nx.path_graph(range(1, 7))
    particiones = {1: [1, 2, 5, 6], 2: [3, 4]}

    cortes = find_connectivity_cuts(g, particiones)

    # un corte por componente, y solo para la partición 1
    assert len(cortes) == 2
    assert all(c[0] == 1 for c in cortes)
    for corte in cortes:
        _revisar_corte(g, particiones, corte)


def test_cuts_varias_componentes():
    # Estrella con centro 0: si el centro queda fuera, cada hoja es una componente
    g = nx.star_graph(4)
    particiones = {1: [1, 2, 3], 2: [0, 4]}

    cortes = find_connectivity_cuts(g, particiones)

    assert len(cortes) == 3
    for corte in cortes:
        _revisar_corte(g, particiones, corte)
        assert corte[3] == {0}


def test_cuts_separador_es_punto_de_articulacion():
    # Dos K4 unidos por un camino de 2 nodos; tomar ambos cliques sin el camino
    g = nx.barbell_graph(4, 2)
    cliques = [0, 1, 2, 3, 6, 7, 8, 9]
    particiones = {1: cliques, 2: [4, 5]}

    cortes = find_connectivity_cuts(g, particiones)

    assert len(cortes) == 2
    for corte in cortes:
        _revisar_corte(g, particiones, corte)
        # basta con uno de los nodos del camino
        assert len(corte[3]) == 1 and corte[3] <= {4, 5}


@pytest.mark.parametrize("semilla", range(20))
def test_cuts_asignaciones_aleatorias(semilla):
    """Todos los cortes generados deben ser válidos en asignaciones al azar."""
    rng = random.Random(semilla)
    n = rng.randint(8, 20)
    g = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=semilla)
    g = nx.relabel_nodes(g, {i: i + 1 for i in g.nodes()})
    k = rng.randint(2, 4)

    particiones = {p: [] for p in range(1, k + 1)}
    for i in g.nodes():
        particiones[rng.randint(1, k)].append(i)

    cortes = find_connectivity_cuts(g, particiones)

    for corte in cortes:
        _revisar_corte(g, particiones, corte)

    # debe haber cortes justo en las particiones que no son conexas
    desconectadas = {
        p for p, nodos in particiones.items()
        if len(nodos) > 1 and not nx.is_connected(g.subgraph(nodos))
    }
    assert {c[0] for c in cortes} == desconectadas


@requiere_gurobi
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


# ---------------------------------------------------------------------------
# Pruebas de estrés: grafos con puntos de articulación (cuellos de botella)
# ---------------------------------------------------------------------------

def _cadena_de_triangulos():
    # triángulos (0,1,2), (2,3,4), (4,5,6): los nodos 2 y 4 son de articulación
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6), (4, 6)])
    return g


def _peine():
    # columna 0-1-2-3 con un diente colgando de cada nodo
    g = nx.path_graph(4)
    g.add_edges_from((i, i + 4) for i in range(4))
    return g


def _dos_estrellas():
    g = nx.star_graph(3)
    g.add_edges_from([(0, 10), (10, 11), (10, 12), (10, 13)])
    return g


GRAFOS_ARTICULACION = {
    "barbell_camino": lambda: nx.barbell_graph(3, 3),
    "lollipop": lambda: nx.lollipop_graph(4, 4),
    "cadena_triangulos": _cadena_de_triangulos,
    "peine": _peine,
    "arbol_binario": lambda: nx.balanced_tree(2, 2),
    "dos_estrellas": _dos_estrellas,
}

SOLVERS = {
    "gurobi": solve_bcp_gurobi if HAY_GUROBI else None,
    "cplex": solve_bcp_cplex if HAY_CPLEX else None,
}


def _optimo_fuerza_bruta(g, k, epsilon):
    """Óptimo exacto enumerando todas las asignaciones (solo sirve en grafos chicos).

    Devuelve None si no existe partición balanceada y conexa.
    """
    nodos = list(g.nodes())
    n = len(nodos)
    l_min = max(1, math.floor((1 - epsilon) * n / k + 1e-9))
    u_max = math.ceil((1 + epsilon) * n / k - 1e-9)

    mejor = None
    # el primer nodo se fija en el grupo 0 para no repetir permutaciones
    for resto in itertools.product(range(k), repeat=n - 1):
        grupo = dict(zip(nodos, (0,) + resto))
        bloques = [[v for v in nodos if grupo[v] == p] for p in range(k)]
        if any(not (l_min <= len(b) <= u_max) for b in bloques):
            continue
        if any(not nx.is_connected(g.subgraph(b)) for b in bloques):
            continue
        cortadas = sum(1 for u, v in g.edges() if grupo[u] != grupo[v])
        if mejor is None or cortadas < mejor:
            mejor = cortadas
    return mejor


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("nombre", GRAFOS_ARTICULACION)
@pytest.mark.parametrize(("k", "epsilon"), [(2, 0.0), (3, 0.0), (3, 0.2)])
def test_estres_articulacion(solver, nombre, k, epsilon):
    """Ninguna solución final puede tener particiones desconectadas."""
    if SOLVERS[solver] is None:
        pytest.skip(f"{solver} no instalado")

    g = GRAFOS_ARTICULACION[nombre]()
    esperado = _optimo_fuerza_bruta(g, k, epsilon)
    res = SOLVERS[solver](g, k=k, epsilon=epsilon, verbose=False, time_limit=60)

    if esperado is None:
        assert not res["has_solution"]
        assert res["status"] in ("INFEASIBLE", "INF_OR_UNBD")
        return

    assert res["status"] == "OPTIMAL"
    assert res["objective"] == esperado
    assert res["is_connected"] is True
    for bloque in res["partitions"].values():
        assert nx.is_connected(g.subgraph(bloque))

    sizes = res["balance_stats"]["sizes"].values()
    assert all(res["balance_stats"]["l_min"] <= s <= res["balance_stats"]["u_max"] for s in sizes)
