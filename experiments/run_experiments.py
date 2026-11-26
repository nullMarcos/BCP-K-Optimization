"""
Ejecuta el ciclo experimental por solver (Gurobi / CPLEX) e instancia,
guardando archivos .log en `outputs/logs/` y exportando métricas a CSV en `outputs/results/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import pandas as pd

from src.instance_loader import load_instances
from src.utils import compute_graph_metrics, solve_bcp

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_solver_fn(solver_name: str) -> Callable[..., Dict[str, Any]]:
    """Retorna un callable que invoca `src.utils.solve_bcp` con el solver indicado ('gurobi' o 'cplex')."""
    name = solver_name.strip().lower()
    if name not in ("gurobi", "cplex"):
        raise ValueError(f"Solver no soportado: {solver_name}")
    return lambda **kwargs: solve_bcp(solver=name, **kwargs)


def _extract_balance_metrics(
    res: Mapping[str, Any],
    num_nodes: int,
    k: int,
) -> Dict[str, Any]:
    """Extrae cotas teóricas [l_min, u_max], tamaños reales de bloques y desbalance efectivo observado."""
    balance_stats = res.get("balance_stats") or {}
    l_min = balance_stats.get("l_min")
    u_max = balance_stats.get("u_max")
    sizes_map = balance_stats.get("sizes") or {}
    sizes = [int(s) for s in sizes_map.values() if s is not None]
    has_solution = bool(res.get("has_solution", res.get("objective") is not None))

    if has_solution and sizes and any(s > 0 for s in sizes) and num_nodes > 0 and k > 0:
        min_size = min(sizes)
        max_size = max(sizes)
        ideal = num_nodes / float(k)
        observed_imbalance = round(max(abs(s - ideal) / ideal for s in sizes), 4)
    else:
        min_size = None
        max_size = None
        observed_imbalance = None

    return {
        "l_min": l_min,
        "u_max": u_max,
        "min_block_size": min_size,
        "max_block_size": max_size,
        "observed_imbalance": observed_imbalance,
    }


def run_experiments(
    instances: Optional[Mapping[str, nx.Graph]] = None,
    solvers: Sequence[str] = ("gurobi", "cplex"),
    k_values: Sequence[int] = (2,),
    epsilons: Sequence[float] = (0.05,),
    time_limit: float = 600.0,
    threads: Optional[int] = None,
    use_symmetry_breaking: bool = True,
    use_lazy_connectivity: bool = True,
    output_csv: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Ejecuta el ciclo experimental, guarda los `.log` y exporta los resultados a CSV en una carpeta por fecha."""
    instances = instances or load_instances()
    run_dir = DEFAULT_OUTPUTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = Path(output_csv) if output_csv else run_dir / "experiment_results.csv"
    logs_dir = Path(logs_dir) if logs_dir else run_dir / "logs"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    r6_suffix = "" if use_symmetry_breaking else "_noR6"

    for inst_name, graph in instances.items():
        m = compute_graph_metrics(graph)

        for k in k_values:
            if m["num_nodes"] < k:
                continue

            for eps in epsilons:
                for solver in solvers:
                    s_norm = solver.strip().lower()
                    log_name = f"{inst_name}_k{k}_eps{int(round(eps * 100)):03d}{r6_suffix}.log"
                    log_file = logs_dir / s_norm / log_name
                    log_file.parent.mkdir(parents=True, exist_ok=True)

                    r6_tag = "R6=ON" if use_symmetry_breaking else "R6=OFF"
                    print(
                        f"Ejecutando {inst_name} (n={m['num_nodes']}, m={m['num_edges']}) | "
                        f"solver={s_norm.upper()} | k={k} | eps={eps:.2f} | {r6_tag} ...",
                        end=" ",
                        flush=True,
                    )

                    res: Dict[str, Any] = {"status": "ERROR"}
                    try:
                        solver_kwargs: Dict[str, Any] = {
                            "graph": graph,
                            "k": k,
                            "epsilon": eps,
                            "time_limit": time_limit,
                            "use_symmetry_breaking": use_symmetry_breaking,
                            "use_lazy_connectivity": use_lazy_connectivity,
                            "log_file": str(log_file),
                            "verbose": False,
                        }
                        if threads is not None:
                            solver_kwargs["threads"] = threads

                        res = get_solver_fn(s_norm)(**solver_kwargs)
                        if res.get("model") is not None and hasattr(res["model"], "dispose"):
                            res["model"].dispose()
                        print(f"-> {res.get('status')} | Z*={res.get('objective')} | t={res.get('runtime', 0.0):.3f}s")
                        if res.get("status") == "INTERRUPTED":
                            print("\n[Ejecución interrumpida por el usuario (Ctrl+C)]")
                            return pd.DataFrame(records)
                    except KeyboardInterrupt:
                        print("\n[Ejecución interrumpida por el usuario (Ctrl+C)]")
                        return pd.DataFrame(records)
                    except Exception as exc:
                        res["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"-> ERROR ({res['error']})")

                    runtime_val = round(float(res["runtime"]), 4) if res.get("runtime") is not None else None
                    wall_time_val = round(float(res["wall_time"]), 4) if res.get("wall_time") is not None else None
                    cb_time_val = (
                        round(float(res["callback_time"]), 4)
                        if res.get("callback_time") is not None
                        else None
                    )
                    if cb_time_val is not None and runtime_val is not None and runtime_val > 0:
                        cb_pct_val: Optional[float] = round(min(100.0, 100.0 * cb_time_val / runtime_val), 2)
                    elif cb_time_val is not None:
                        cb_pct_val = 0.0
                    else:
                        cb_pct_val = None

                    bal_metrics = _extract_balance_metrics(res, m["num_nodes"], k)

                    records.append({
                        "instance": inst_name,
                        "family": graph.graph.get("family", "unknown"),
                        "solver": s_norm.upper(),
                        "k": k,
                        "epsilon": eps,
                        "use_symmetry_breaking": use_symmetry_breaking,
                        "use_lazy_connectivity": use_lazy_connectivity,
                        "threads": threads,
                        **m,
                        "status": res.get("status"),
                        "has_solution": bool(res.get("has_solution", res.get("objective") is not None)),
                        "objective": res.get("objective"),
                        "best_bound": res.get("best_bound"),
                        "mip_gap": res.get("mip_gap"),
                        "runtime": runtime_val,
                        "wall_time": wall_time_val,
                        "callback_time": cb_time_val,
                        "callback_time_pct": cb_pct_val,
                        "node_count": res.get("node_count", 0),
                        "lazy_cuts_added": res.get("lazy_cuts_added", 0),
                        "partition_connected": res.get("is_connected", False),
                        **bal_metrics,
                        "log_file": (
                            str(log_file.relative_to(PROJECT_ROOT))
                            if log_file.resolve().is_relative_to(PROJECT_ROOT)
                            else str(log_file)
                        ),
                        "error": res.get("error"),
                    })
                    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nResultados guardados en: {output_csv}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecución de experimentos BCP_k en Luthier.")
    parser.add_argument("--solvers", nargs="+", default=["gurobi", "cplex"])
    parser.add_argument("--instances", nargs="+", default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.05])
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Número máximo de hilos por solver para comparación justa.",
    )
    parser.add_argument(
        "--no-symmetry-breaking",
        action="store_true",
        help="Desactiva las restricciones de ruptura de simetrías (R6).",
    )
    parser.add_argument(
        "--no-lazy-connectivity",
        action="store_true",
        help="Desactiva las restricciones perezosas de conectividad (R4).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    run_experiments(
        instances=load_instances(names=args.instances),
        solvers=args.solvers,
        k_values=args.k_values,
        epsilons=args.epsilons,
        time_limit=args.time_limit,
        threads=args.threads,
        use_symmetry_breaking=not args.no_symmetry_breaking,
        use_lazy_connectivity=not args.no_lazy_connectivity,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
