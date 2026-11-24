"""
Barridos paramétricos para BCP_k (se ejecutan en el servidor; reutilizan `run_experiments`).

1. `epsilon`     — Sensibilidad a ε ∈ {0.00, 0.05, 0.10, 0.15}: curva de compromiso entre el
                   balance de los bloques y el número de aristas cortadas Z = Σ y_ij.
2. `scalability` — Escalabilidad en n y k ∈ {2, 3, 4, 5}: identifica el punto de quiebre (menor n
                   en que el solver alcanza el límite de tiempo sin cerrar la prueba) y muestra,
                   a partir de los logs, cómo el MIPGap se estanca en el tiempo.

Con `--from-csv` no se resuelve nada: solo se analiza un CSV ya generado (por ejemplo, el
traído desde el servidor con rsync junto con su carpeta outputs/logs/).

Ejemplos:
    python experiments/parameter_sweep.py epsilon --solvers gurobi cplex --time-limit 300
    python experiments/parameter_sweep.py scalability --solvers gurobi cplex --time-limit 600
    python experiments/parameter_sweep.py scalability --from-csv outputs/results/sweep_scalability_<FECHA>.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.generate_profiles import (  # noqa: E402
    CATEGORICAL,
    FIG_DIR,
    RESULTS_DIR,
    SOLVED_STATUSES,
    SOLVER_STYLE,
    TEXT_SECONDARY,
    apply_style,
    drop_errors,
    ordered_solvers,
    plt,
)

LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"


def _k_color(k: int, ks: Sequence[int]) -> str:
    return CATEGORICAL[list(ks).index(k)]


def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["solver"] = df["solver"].astype(str).str.upper()
    df["epsilon"] = df["epsilon"].astype(float).round(4)
    df = drop_errors(df)
    df["closed"] = df["status"].isin(SOLVED_STATUSES)
    return df


def _run(experiment: str, instance_names: Optional[List[str]], solvers: Sequence[str],
         k_values: Sequence[int], epsilons: Sequence[float], time_limit: float) -> Path:
    from experiments.run_experiments import run_experiments
    from src.instance_loader import load_instances

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"sweep_{experiment}_{stamp}.csv"
    run_experiments(
        instances=load_instances(names=instance_names),
        solvers=solvers,
        k_values=k_values,
        epsilons=epsilons,
        time_limit=time_limit,
        output_csv=csv_path,
        logs_dir=LOGS_DIR / f"sweep_{experiment}_{stamp}",
    )
    return csv_path


# ---------------------------------------------------------------------------
# 1. Barrido de ε
# ---------------------------------------------------------------------------
def epsilon_tradeoff_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (solver, instancia, k): Z_rel(ε) = Z(ε) / Z(ε_ref), con ε_ref el menor ε con solución.
    ahorro_cortes_pct = 100·(1 − Z_rel): cuántos cortes se ahorran al relajar el balance.
    """
    keys = ["solver", "instance", "k"]
    refs = (df[df["objective"].notna()].sort_values("epsilon")
            .drop_duplicates(subset=keys, keep="first")[keys + ["objective", "epsilon"]]
            .rename(columns={"objective": "z_ref", "epsilon": "eps_ref"}))
    data = df.merge(refs, on=keys, how="left")
    data["z_rel"] = np.where(data["z_ref"] > 0, data["objective"] / data["z_ref"], np.nan)
    data["ahorro_cortes_pct"] = 100.0 * (1.0 - data["z_rel"])
    cols = keys + ["family", "num_nodes", "epsilon", "status", "closed", "objective", "best_bound",
                   "mip_gap", "runtime", "eps_ref", "z_ref", "z_rel", "ahorro_cortes_pct"]
    return data[cols].sort_values(keys + ["epsilon"]).reset_index(drop=True)


def plot_epsilon_tradeoff(table: pd.DataFrame, out_path: Path) -> Path:
    """Filas = solver, columnas = familia; una línea por k (mediana sobre las instancias de la familia)."""
    families = sorted(table["family"].dropna().unique())
    solvers = ordered_solvers(table["solver"])
    ks = sorted(table["k"].unique())
    eps_values = sorted(table["epsilon"].unique())

    fig, axes = plt.subplots(len(solvers), len(families), figsize=(3.0 * len(families) + 0.6, 2.9 * len(solvers)),
                             sharex=True, sharey=True, squeeze=False)
    for i, solver in enumerate(solvers):
        for j, fam in enumerate(families):
            ax = axes[i][j]
            sub = table[(table["solver"] == solver) & (table["family"] == fam)]
            for k in ks:
                g = sub[sub["k"] == k].groupby("epsilon").agg(z=("z_rel", "median"), closed=("closed", "all"))
                if g["z"].notna().any():
                    color = _k_color(k, ks)
                    ax.plot(g.index, g["z"], color=color, label=f"k={k}", zorder=2)
                    ax.scatter(g.index, g["z"], s=40, zorder=3, linewidth=1.5, edgecolor=color,
                               facecolor=[color if c else "white" for c in g["closed"]])
            ax.set_xticks(eps_values)
            ax.set_xticklabels([f"{e:.2f}" for e in eps_values])
            if i == 0:
                ax.set_title(fam, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{solver}\n" + r"$Z(\varepsilon)\,/\,Z(\varepsilon_{ref})$", fontsize=9)
            if i == len(solvers) - 1:
                ax.set_xlabel(r"tolerancia $\varepsilon$")
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]),
                           ([], []))
    fig.legend(handles, labels, loc="lower center", ncol=len(ks), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, "Marcador hueco: al menos una instancia no cerró la prueba dentro del límite de tiempo.",
             ha="center", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analyze_epsilon(csv_path: Path) -> Dict[str, Path]:
    table = epsilon_tradeoff_table(_read_csv(csv_path))
    table_path = RESULTS_DIR / "epsilon_tradeoff.csv"
    table.to_csv(table_path, index=False, encoding="utf-8")
    return {"tabla": table_path, "figura": plot_epsilon_tradeoff(table, FIG_DIR / "epsilon_tradeoff.png")}


# ---------------------------------------------------------------------------
# 2. Escalabilidad y punto de quiebre
# ---------------------------------------------------------------------------
def breakpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    """Por (solver, k, ε): mayor n cerrado y punto de quiebre (menor n que no cerró la prueba)."""
    rows = []
    for (solver, k, eps), g in df.groupby(["solver", "k", "epsilon"]):
        g = g.sort_values("num_nodes")
        closed, open_ = g[g["closed"]], g[~g["closed"]]
        first = open_.iloc[0] if len(open_) else None
        rows.append({
            "solver": solver, "k": k, "epsilon": eps, "instancias": len(g), "cerradas": len(closed),
            "n_max_cerrado": int(closed["num_nodes"].max()) if len(closed) else None,
            "n_quiebre": int(first["num_nodes"]) if first is not None else None,
            "instancia_quiebre": first["instance"] if first is not None else None,
            "gap_en_quiebre_pct": round(100.0 * float(first["mip_gap"]), 2)
            if first is not None and pd.notna(first["mip_gap"]) else None,
        })
    return pd.DataFrame(rows)


def plot_scalability(df: pd.DataFrame, out_path: Path, time_limit: Optional[float]) -> Path:
    """Filas = solver; (a) tiempo vs n, (b) MIPGap final vs n; una línea por k."""
    solvers = ordered_solvers(df["solver"])
    ks = sorted(df["k"].unique())
    data = df.assign(gap_pct=100.0 * df["mip_gap"].astype(float))

    fig, axes = plt.subplots(len(solvers), 2, figsize=(11, 3.6 * len(solvers)), sharex=True, squeeze=False)
    for i, solver in enumerate(solvers):
        ax_t, ax_g = axes[i]
        for k in ks:
            g = (data[(data["solver"] == solver) & (data["k"] == k)]
                 .groupby("num_nodes").agg(t=("runtime", "median"), gap=("gap_pct", "median"),
                                           closed=("closed", "all")))
            if g.empty:
                continue
            color = _k_color(k, ks)
            faces = [color if c else "white" for c in g["closed"]]
            for ax, col in ((ax_t, "t"), (ax_g, "gap")):
                ax.plot(g.index, g[col], color=color, label=f"k={k}", zorder=2)
                ax.scatter(g.index, g[col], s=40, facecolor=faces, edgecolor=color, linewidth=1.5, zorder=3)
        if time_limit:
            ax_t.axhline(time_limit, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
            ax_t.annotate("límite de tiempo", xy=(0.0, time_limit), xycoords=("axes fraction", "data"),
                          xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
        ax_t.set_yscale("log")
        ax_t.set_ylabel(f"{solver}\ntiempo [s]")
        ax_g.set_ylabel("MIPGap final [%]")
        if i == 0:
            ax_t.set_title("(a) Tiempo de resolución vs n")
            ax_g.set_title("(b) Brecha de optimalidad vs n")
    for ax in axes[-1]:
        ax.set_xlabel("número de nodos n")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=len(ks), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, "Marcador hueco: el solver no cerró la prueba (límite de tiempo alcanzado).",
             ha="center", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2b. Estancamiento del MIPGap en el tiempo (desde los logs de los solvers)
# ---------------------------------------------------------------------------
# Gurobi: "  7259   649 infeasible   12        18.00000   13.15197  26.9%  92.8   10s"
_GUROBI_ROW = re.compile(r"\s(\S+)\s+(\S+)\s+([\d.]+)%\s+\S+\s+(\d+)s\s*$")
# CPLEX: las filas del árbol terminan en "52.00%" y el tiempo aparece aparte en "Elapsed time = 12.34 sec."
_CPLEX_ROW = re.compile(r"\s([\d.]+)%\s*$")
_CPLEX_TIME = re.compile(r"Elapsed time\s*=\s*([\d.]+)\s*sec")


def parse_gap_trace(log_path: Path, solver: str) -> pd.DataFrame:
    """
    Evolución (tiempo [s], gap [%]) leída del log del solver.
    Resolución temporal: 1 s en Gurobi; en CPLEX cada gap toma el último "Elapsed time" impreso.
    """
    points = []
    elapsed = 0.0
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if solver == "GUROBI":
                m = _GUROBI_ROW.search(line)
                if m:
                    points.append((float(m.group(4)), float(m.group(3))))
            else:
                mt = _CPLEX_TIME.search(line)
                if mt:
                    elapsed = float(mt.group(1))
                    continue
                m = _CPLEX_ROW.search(line)
                if m:
                    points.append((elapsed, float(m.group(1))))
    return pd.DataFrame(points, columns=["time", "gap_pct"])


def plot_gap_vs_time(df: pd.DataFrame, out_path: Path, time_limit: Optional[float]) -> Optional[Path]:
    """Filas = solver, columnas = k; una curva por instancia que alcanzó el límite de tiempo."""
    runs = df[~df["closed"] & df["log_file"].notna()]
    traces = []
    for _, row in runs.iterrows():
        path = PROJECT_ROOT / row["log_file"]
        if path.exists():
            tr = parse_gap_trace(path, row["solver"])
            if len(tr):
                traces.append((row, tr))
    if not traces:
        print("[aviso] No hay logs con trayectoria de gap para corridas no cerradas: se omite gap vs tiempo.")
        return None

    solvers = ordered_solvers(r["solver"] for r, _ in traces)
    ks = sorted({r["k"] for r, _ in traces})
    fig, axes = plt.subplots(len(solvers), len(ks), figsize=(3.4 * len(ks) + 0.6, 3.0 * len(solvers)),
                             sharex=True, sharey=True, squeeze=False)
    for row, tr in traces:
        ax = axes[solvers.index(row["solver"])][ks.index(row["k"])]
        ax.step(tr["time"], tr["gap_pct"], where="post", linewidth=1.5, alpha=0.8,
                color=SOLVER_STYLE.get(row["solver"], {"color": CATEGORICAL[2]})["color"])
        ax.annotate(f"{row['instance']} (n={row['num_nodes']})", xy=(tr["time"].iloc[-1], tr["gap_pct"].iloc[-1]),
                    xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=7, color=TEXT_SECONDARY)
    for i, solver in enumerate(solvers):
        axes[i][0].set_ylabel(f"{solver}\nMIPGap [%]")
    for j, k in enumerate(ks):
        axes[0][j].set_title(f"k={k}", fontsize=10)
        axes[-1][j].set_xlabel("tiempo [s]")
        if time_limit:
            for i in range(len(solvers)):
                axes[i][j].axvline(time_limit, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
    fig.suptitle("Estancamiento del MIPGap en corridas que alcanzan el límite de tiempo", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "Resolución temporal: 1 s en Gurobi; en CPLEX, la del último 'Elapsed time' del log.",
             ha="center", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analyze_scalability(csv_path: Path, time_limit: Optional[float]) -> Dict[str, Path]:
    df = _read_csv(csv_path)
    bp = breakpoint_table(df)
    bp_path = RESULTS_DIR / "scalability_breakpoints.csv"
    bp.to_csv(bp_path, index=False, encoding="utf-8")
    print(bp.to_string(index=False))
    outputs = {"quiebres": bp_path, "figura": plot_scalability(df, FIG_DIR / "scalability.png", time_limit)}
    gap_fig = plot_gap_vs_time(df, FIG_DIR / "gap_vs_time.png", time_limit)
    if gap_fig:
        outputs["gap_vs_tiempo"] = gap_fig
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Barridos paramétricos de BCP_k (ε y escalabilidad).")
    sub = parser.add_subparsers(dest="experiment", required=True)
    for name, k_default, eps_default, tl_default in [
        ("epsilon", [2, 3], [0.00, 0.05, 0.10, 0.15], 300.0),
        ("scalability", [2, 3, 4, 5], [0.05], 600.0),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--solvers", nargs="+", default=["gurobi", "cplex"])
        p.add_argument("--instances", nargs="+", default=None, help="Nombres de instancias (por defecto todas)")
        p.add_argument("--k-values", nargs="+", type=int, default=k_default)
        p.add_argument("--epsilons", nargs="+", type=float, default=eps_default)
        p.add_argument("--time-limit", type=float, default=tl_default)
        p.add_argument("--from-csv", type=Path, default=None, help="Solo analiza un CSV existente, sin resolver")
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = args.from_csv or _run(args.experiment, args.instances, args.solvers,
                                     args.k_values, args.epsilons, args.time_limit)
    outputs = (analyze_epsilon(csv_path) if args.experiment == "epsilon"
               else analyze_scalability(csv_path, args.time_limit))
    for key, path in outputs.items():
        print(f"  {key:14s} -> {path}")


if __name__ == "__main__":
    main()
