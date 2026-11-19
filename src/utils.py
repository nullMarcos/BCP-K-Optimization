"""
Módulo de utilidades generales, preprocesamiento y métricas topológicas para BCP_k.

Contenido principal:
    - Preprocesamiento y validación de grafos.
    - Extracción de métricas topológicas (densidad, grado promedio, transitividad, etc.).
    - Utilidades para manejo de datos de instancias.

Nota de integración:
    - La gestión de licencias se encuentra modularizada en `src.license` y se
      re-exporta aquí para mantener compatibilidad hacia atrás.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import networkx as nx

# Re-exportación de configuración de licencias (modularizado en src.license)
from src.license import configurar_licencia

__all__ = [
    "configurar_licencia",
]

# ============================================================================
# Sección reservada para Francisco: Métricas Topológicas y Preprocesamiento
# ============================================================================

def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calcula métricas topológicas clave para una instancia de grafo.

    Args:
        graph: Grafo de NetworkX.

    Returns:
        Dict con métricas como número de nodos, número de aristas, densidad,
        grado promedio, conectividad algebraica, etc.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = nx.density(graph) if n > 1 else 0.0
    avg_degree = (2.0 * m) / n if n > 0 else 0.0

    return {
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "is_connected": nx.is_connected(graph) if n > 0 else False,
    }
