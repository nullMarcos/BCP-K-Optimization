"""
Perfiles de desempeño de Dolan & Moré (2002), análisis de actividad Branch-and-Cut
y gráficos comparativos Gurobi vs CPLEX.

Sea P el conjunto de problemas (instancia, k, epsilon) y S el conjunto de solvers.
t_{p,s} es el tiempo que el solver s necesita para resolver p. La razón de desempeño es

    r_{p,s} = t_{p,s} / min{ t_{p,s'} : s' ∈ S }

con r_{p,s} = ∞ si s no resolvió p (no demostró optimalidad ni infactibilidad dentro
del límite de tiempo). La función acumulada de desempeño del solver s es

    ρ_s(τ) = |{ p ∈ P : r_{p,s} ≤ τ }| / |P|

ρ_s(1) es la fracción de problemas en que s fue el más rápido, y ρ_s(τ) para τ grande
es la fracción de problemas que s logra resolver.

Uso:
    python experiments/generate_profiles.py outputs/results/sweep_*.csv --time-limit 600
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

PROBLEM_COLS = ["instance", "k", "epsilon"]
# Resuelto = el solver cerró la prueba (óptimo o infactibilidad demostrada).
SOLVED_STATUSES = ("OPTIMAL", "INFEASIBLE")
# Piso de tiempo: evita razones artificiales entre corridas de milisegundos (0.001 s vs 0.004 s → r = 4).
MIN_RUNTIME = 0.01

# Colores fijos por entidad (paleta categórica validada para daltonismo): Gurobi azul, CPLEX naranjo.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SOLVER_STYLE = {
    "GUROBI": {"color": CATEGORICAL[0], "linestyle": "-", "marker": "o"},
    "CPLEX": {"color": CATEGORICAL[1], "linestyle": "--", "marker": "s"},
}
TEXT_SECONDARY = "#52514e"


def apply_style() -> None:
    plt.rcParams.update({
        "axes.edgecolor": "#e4e3df", "axes.labelcolor": TEXT_SECONDARY, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e4e3df", "grid.linewidth": 0.8, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "legend.frameon": False, "lines.linewidth": 2.0,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def ordered_solvers(solvers: Iterable[str]) -> List[str]:
    known = list(SOLVER_STYLE)
    return sorted(set(solvers), key=lambda s: (known.index(s) if s in known else len(known), s))


def drop_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Excluye corridas que terminaron en ERROR: un fallo de ejecución no es un timeout."""
    is_error = df["status"].isna() | df["status"].astype(str).str.upper().eq("ERROR")
    if is_error.any():
        print(f"[aviso] Se excluyen {int(is_error.sum())} corridas con ERROR:")
        for _, row in df[is_error].iterrows():
            print(f"    {row['solver']} {row['instance']} k={row['k']} eps={row['epsilon']}: {row.get('error')}")
    return df[~is_error].copy()


def load_results(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Une CSV de `run_experiments.py`; si un problema se repite, conserva la última corrida."""
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df["solver"] = df["solver"].astype(str).str.upper()
    df["epsilon"] = df["epsilon"].astype(float).round(4)
    df = drop_errors(df)
    dedup_cols = list(PROBLEM_COLS) + ["solver"]
    if "use_symmetry_breaking" in df.columns:
        dedup_cols.append("use_symmetry_breaking")
    return df.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Perfiles de desempeño
# ---------------------------------------------------------------------------
def cost_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """t_{p,s}: filas = problemas, columnas = solvers; ∞ si no resolvió. Solo problemas corridos por todos."""
    data = df.copy()
    if "use_symmetry_breaking" in data.columns:
        data = data[data["use_symmetry_breaking"].fillna(True).astype(bool)]
    solved = data["status"].isin(SOLVED_STATUSES) & data["runtime"].notna()
    data["cost"] = np.where(solved, data["runtime"].clip(lower=MIN_RUNTIME), np.inf)
    t = data.pivot_table(index=PROBLEM_COLS, columns="solver", values="cost", aggfunc="first")
    return t.dropna(how="any")


def performance_ratios(t: pd.DataFrame) -> pd.DataFrame:
    """r_{p,s} = t_{p,s} / min_s t_{p,s}; si ningún solver resolvió p, r = ∞ para todos."""
    best = t.min(axis=1)
    r = t.div(best, axis=0)
    r.loc[~np.isfinite(best)] = np.inf
    return r


def performance_profile(r: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    """ρ_s(τ) para cada τ (filas) y solver (columnas)."""
    return pd.DataFrame({s: [(r[s] <= tau).mean() for tau in taus] for s in r.columns},
                        index=pd.Index(taus, name="tau"))


def tau_grid(r: pd.DataFrame, n_points: int = 400) -> np.ndarray:
    """Grilla log₂ desde τ = 1, incluyendo las razones exactas para que los saltos sean exactos."""
    finite = r.to_numpy()[np.isfinite(r.to_numpy())]
    tau_max = max(2.0, float(finite.max()) * 1.1) if finite.size else 2.0
    return np.unique(np.concatenate([np.logspace(0, np.log2(tau_max), n_points, base=2.0), finite]))


def plot_performance_profile(rho: pd.DataFrame, n_problems: int, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for solver in ordered_solvers(rho.columns):
        st = SOLVER_STYLE.get(solver, {"color": CATEGORICAL[2], "linestyle": "-"})
        ax.step(rho.index, rho[solver], where="post", color=st["color"], linestyle=st["linestyle"],
                label=f"{solver} (resuelve {rho[solver].iloc[-1]:.0%})")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, rho.index.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\tau$ (factor respecto al mejor solver, escala log$_2$)")
    ax.set_ylabel(r"$\rho_s(\tau)$")
    ax.set_title(f"Perfil de desempeño Dolan–Moré: tiempo de resolución (|P| = {n_problems})")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Boxplots de tiempo y diagrama de gap
# ---------------------------------------------------------------------------
def _grouped_boxplot(ax, groups: Sequence, solvers: Sequence[str], values_of) -> None:
    width = 0.8 / max(1, len(solvers))
    for j, solver in enumerate(solvers):
        color = SOLVER_STYLE.get(solver, {"color": CATEGORICAL[2]})["color"]
        pos = np.arange(len(groups)) + (j - (len(solvers) - 1) / 2) * width
        series_list = [np.asarray(values_of(solver, g), dtype=float) for g in groups]
        non_empty = [s[~np.isnan(s)] if len(s) else np.array([np.nan]) for s in series_list]
        bp = ax.boxplot(non_empty, positions=pos, widths=width * 0.85,
                        patch_artist=True, medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": color,
                                    "markeredgecolor": color})
        for patch in bp["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.75)
        ax.plot([], [], color=color, linewidth=6, alpha=0.75, label=solver)
    ax.set_xticks(np.arange(len(groups)))


def plot_runtime_boxplots(df: pd.DataFrame, out_path: Path, time_limit: Optional[float]) -> Path:
    families = sorted(df["family"].dropna().unique())
    fig, ax = plt.subplots(figsize=(max(6.4, 1.3 * len(families) + 2), 4.2))
    _grouped_boxplot(ax, families, ordered_solvers(df["solver"]),
                     lambda s, f: df[(df["family"] == f) & (df["solver"] == s)]["runtime"].dropna()
                     .clip(lower=MIN_RUNTIME))
    if time_limit:
        ax.axhline(time_limit, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
        ax.annotate("límite de tiempo", xy=(1.0, time_limit), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylabel("tiempo de resolución [s]")
    ax.set_title("Tiempo de resolución por familia de instancias")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_gap_diagram(df: pd.DataFrame, out_path: Path) -> Path:
    """(a) MIPGap final vs n; (b) MIPGap final por k."""
    data = df[df["mip_gap"].notna()].copy()
    data["gap_pct"] = 100.0 * data["mip_gap"].astype(float)
    solvers = ordered_solvers(data["solver"])
    ks = sorted(data["k"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for solver in solvers:
        st = SOLVER_STYLE.get(solver, {"color": CATEGORICAL[2], "marker": "^"})
        g = data[data["solver"] == solver]
        ax1.scatter(g["num_nodes"], g["gap_pct"], s=36, marker=st["marker"], facecolor=st["color"],
                    edgecolor="white", linewidth=1.0, alpha=0.85, label=solver, zorder=3)
    ax1.set_xlabel("número de nodos n")
    ax1.set_title("(a) Brecha de optimalidad vs tamaño")
    ax1.legend(loc="upper left")

    _grouped_boxplot(ax2, ks, solvers, lambda s, k: data[(data["solver"] == s) & (data["k"] == k)]["gap_pct"])
    ax2.set_xticklabels([f"k={k}" for k in ks])
    ax2.set_title("(b) Brecha de optimalidad por k")

    top = max(5.0, 1.1 * float(data["gap_pct"].max())) if len(data) else 5.0
    for ax in (ax1, ax2):
        ax.set_ylim(-0.03 * top, top)
        ax.set_ylabel("MIPGap final [%]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Actividad de Branch-and-Cut (cortes perezosos R4, nodos B&B y tiempo en callback)
# ---------------------------------------------------------------------------
def branch_and_cut_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume por (solver, family, k) el esfuerzo del esquema Branch-and-Cut:
    cortes de conectividad R4 inyectados, nodos explorados y tiempo en callback.
    """
    data = df.copy()
    if "callback_time" not in data.columns:
        data["callback_time"] = np.nan
    if "callback_time_pct" not in data.columns:
        valid_rt = data["runtime"].notna() & (data["runtime"] > 0) & data["callback_time"].notna()
        data["callback_time_pct"] = np.where(
            valid_rt,
            np.minimum(100.0, 100.0 * data["callback_time"] / data["runtime"]),
            np.nan,
        )

    rows = []
    for (solver, family, k), g in data.groupby(["solver", "family", "k"], dropna=False):
        rows.append({
            "solver": solver,
            "family": family,
            "k": int(k),
            "corridas": len(g),
            "cerradas": int(g["status"].isin(SOLVED_STATUSES).sum()),
            "median_lazy_cuts": round(float(g["lazy_cuts_added"].median()), 2)
            if "lazy_cuts_added" in g and g["lazy_cuts_added"].notna().any() else None,
            "mean_lazy_cuts": round(float(g["lazy_cuts_added"].mean()), 2)
            if "lazy_cuts_added" in g and g["lazy_cuts_added"].notna().any() else None,
            "max_lazy_cuts": int(g["lazy_cuts_added"].max())
            if "lazy_cuts_added" in g and g["lazy_cuts_added"].notna().any() else None,
            "median_node_count": round(float(g["node_count"].median()), 2)
            if "node_count" in g and g["node_count"].notna().any() else None,
            "mean_node_count": round(float(g["node_count"].mean()), 2)
            if "node_count" in g and g["node_count"].notna().any() else None,
            "median_runtime_s": round(float(g["runtime"].median()), 4)
            if "runtime" in g and g["runtime"].notna().any() else None,
            "median_callback_time_s": round(float(g["callback_time"].median()), 4)
            if g["callback_time"].notna().any() else None,
            "median_callback_time_pct": round(float(g["callback_time_pct"].median()), 2)
            if g["callback_time_pct"].notna().any() else None,
        })
    return pd.DataFrame(rows).sort_values(["solver", "family", "k"]).reset_index(drop=True)


def plot_branch_and_cut_activity(df: pd.DataFrame, out_path: Path) -> Path:
    """
    Grafica por familia topológica y solver:
      (a) Cortes perezosos R4 inyectados (lazy_cuts_added)
      (b) Nodos explorados en el árbol Branch-and-Bound (node_count)
      (c) Porcentaje de tiempo consumido dentro del callback R4 (si está disponible)
    """
    families = sorted(df["family"].dropna().unique())
    solvers = ordered_solvers(df["solver"])
    has_cb_pct = "callback_time_pct" in df.columns and df["callback_time_pct"].notna().any()
    n_panels = 3 if has_cb_pct else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.2))
    if n_panels == 2:
        ax_cuts, ax_nodes = axes
        ax_cb = None
    else:
        ax_cuts, ax_nodes, ax_cb = axes

    _grouped_boxplot(
        ax_cuts,
        families,
        solvers,
        lambda s, f: df[(df["family"] == f) & (df["solver"] == s)]["lazy_cuts_added"].dropna(),
    )
    ax_cuts.set_xticklabels(families, rotation=20, ha="right")
    ax_cuts.set_ylabel("cortes perezosos R4 inyectados")
    ax_cuts.set_title("(a) Cortes de conectividad (R4)")
    ax_cuts.legend(loc="upper left")

    _grouped_boxplot(
        ax_nodes,
        families,
        solvers,
        lambda s, f: df[(df["family"] == f) & (df["solver"] == s)]["node_count"].dropna().clip(lower=1),
    )
    ax_nodes.set_yscale("log")
    ax_nodes.set_xticklabels(families, rotation=20, ha="right")
    ax_nodes.set_ylabel("nodos B&B explorados (escala log, mín=1)")
    ax_nodes.set_title("(b) Tamaño del árbol Branch-and-Bound")

    if ax_cb is not None:
        _grouped_boxplot(
            ax_cb,
            families,
            solvers,
            lambda s, f: df[(df["family"] == f) & (df["solver"] == s)]["callback_time_pct"].dropna(),
        )
        ax_cb.set_xticklabels(families, rotation=20, ha="right")
        ax_cb.set_ylabel("tiempo en callback / tiempo total [%]")
        ax_cb.set_title("(c) Costo relativo del separador R4")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all(csv_paths: Sequence[Path], time_limit: Optional[float] = None) -> Dict[str, Path]:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results(csv_paths)
    outputs: Dict[str, Path] = {}

    if df["solver"].nunique() >= 2:
        r = performance_ratios(cost_matrix(df))
        if len(r):
            r.to_csv(RESULTS_DIR / "performance_ratios.csv")
            rho = performance_profile(r, tau_grid(r))
            outputs["dolan_more"] = plot_performance_profile(rho, len(r), FIG_DIR / "perfil_dolan_more_tiempo.png")
        else:
            print("[aviso] No hay problemas corridos por todos los solvers: se omite el perfil.")
    else:
        print("[aviso] Solo hay un solver en los datos: se omite el perfil de Dolan–Moré.")

    outputs["boxplot_runtime"] = plot_runtime_boxplots(df, FIG_DIR / "boxplot_tiempo.png", time_limit)
    outputs["gap_diagram"] = plot_gap_diagram(df, FIG_DIR / "diagrama_gap.png")

    bc_table = branch_and_cut_summary(df)
    bc_table_path = RESULTS_DIR / "branch_and_cut_summary.csv"
    bc_table.to_csv(bc_table_path, index=False, encoding="utf-8")
    outputs["bc_summary_csv"] = bc_table_path
    outputs["bc_activity_fig"] = plot_branch_and_cut_activity(df, FIG_DIR / "actividad_branch_and_cut.png")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfiles de Dolan–Moré, boxplots de tiempo y diagrama de gap.")
    parser.add_argument("csv", nargs="+", type=Path, help="CSV(s) generados por run_experiments.py")
    parser.add_argument("--time-limit", type=float, default=None, help="Dibuja la línea del límite de tiempo")
    args = parser.parse_args()
    for name, path in generate_all(args.csv, args.time_limit).items():
        print(f"  {name:16s} -> {path}")


if __name__ == "__main__":
    main()
