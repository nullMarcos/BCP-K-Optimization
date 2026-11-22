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
from src.utils import compute_graph_metrics

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_solver_fn(solver_name: str) -> Callable[..., Dict[str, Any]]:
    """Retorna la función unificada `solve_bcp` del solver indicado ('gurobi' o 'cplex')."""
    name = solver_name.strip().lower()
    if name == "gurobi":
        from src.model_gurobi import solve_bcp

        return solve_bcp
    if name == "cplex":
        from src.model_cplex import solve_bcp

        return solve_bcp
    raise ValueError(f"Solver no soportado: {solver_name}")


def run_experiments(
    instances: Optional[Mapping[str, nx.Graph]] = None,
    solvers: Sequence[str] = ("gurobi", "cplex"),
    k_values: Sequence[int] = (2,),
    epsilons: Sequence[float] = (0.05,),
    time_limit: float = 600.0,
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

    for inst_name, graph in instances.items():
        m = compute_graph_metrics(graph)

        for k in k_values:
            if m["num_nodes"] < k:
                continue

            for eps in epsilons:
                for solver in solvers:
                    s_norm = solver.strip().lower()
                    log_file = logs_dir / s_norm / f"{inst_name}_k{k}_eps{int(round(eps * 100)):03d}.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)

                    print(
                        f"Ejecutando {inst_name} (n={m['num_nodes']}, m={m['num_edges']}) | "
                        f"solver={s_norm.upper()} | k={k} | eps={eps:.2f} ...",
                        end=" ",
                        flush=True,
                    )

                    res: Dict[str, Any] = {"status": "ERROR"}
                    try:
                        res = get_solver_fn(s_norm)(
                            graph=graph,
                            k=k,
                            epsilon=eps,
                            time_limit=time_limit,
                            log_file=str(log_file),
                            verbose=False,
                        )
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

                    records.append({
                        "instance": inst_name,
                        "family": graph.graph.get("family", "unknown"),
                        "solver": s_norm.upper(),
                        "k": k,
                        "epsilon": eps,
                        **m,
                        "status": res.get("status"),
                        "objective": res.get("objective"),
                        "best_bound": res.get("best_bound"),
                        "mip_gap": res.get("mip_gap"),
                        "runtime": round(float(res["runtime"]), 4) if res.get("runtime") is not None else None,
                        "node_count": res.get("node_count", 0),
                        "lazy_cuts_added": res.get("lazy_cuts_added", 0),
                        "partition_connected": res.get("is_connected", False),
                        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
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
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    run_experiments(
        instances=load_instances(names=args.instances),
        solvers=args.solvers,
        k_values=args.k_values,
        epsilons=args.epsilons,
        time_limit=args.time_limit,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
