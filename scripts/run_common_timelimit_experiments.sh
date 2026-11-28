#!/usr/bin/env bash
# ==============================================================================
# Script de ejecución para las instancias que alcanzaron TIME_LIMIT en común
# entre Gurobi y CPLEX (`er_n100_p006` con k ∈ {3, 4} y ε ∈ {0.00, 0.05, 0.10, 0.15}),
# con un límite de tiempo por defecto de 600 segundos por corrida.
#
# Basado en `scripts/run_main_experiments.sh`.
# Guarda su propio CSV en `outputs/results/common_timelimit_600s_*.csv` sin
# sobrescribir `main_experiments_latest.csv`.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"
HOST_TAG="$(hostname 2>/dev/null || echo 'servidor')"
START_EPOCH="$(date +%s)"

# ==============================================================================
# Notificaciones Discord y manejo de errores/interrupciones
# ==============================================================================
format_elapsed() {
    local secs=$(( $(date +%s) - START_EPOCH ))
    printf "%02dh:%02dm:%02ds" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

notify_discord() {
    local title="$1"
    local description="$2"
    local color="$3"  # Decimal: verde=3066993, rojo=15158332, amarillo=16753920, azul=3447003

    if [[ -z "${DISCORD_WEBHOOK_URL}" ]]; then
        return 0
    fi

    local payload
    payload=$(python3 -c '
import json, sys
title, desc, color = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
    "embeds": [{
        "title": title,
        "description": desc,
        "color": color
    }]
}))
' "$title" "$description" "$color" 2>/dev/null || true)

    if [[ -n "$payload" ]]; then
        curl -fsSL -X POST -H "Content-Type: application/json" \
            -d "$payload" \
            "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

on_error() {
    local exit_code="$?"
    local line_no="${1:-unknown}"
    local elapsed
    elapsed="$(format_elapsed)"
    echo ""
    echo "[ERROR] $(basename "$0") falló en la línea ${line_no} (código: ${exit_code}, tiempo: ${elapsed})." >&2
    notify_discord \
        "❌ Fallo en $(basename "$0") (${HOST_TAG})" \
        "Host: \`${HOST_TAG}\`\nLínea: \`${line_no}\` | Exit code: \`${exit_code}\`\nTiempo transcurrido: \`${elapsed}\`" \
        15158332
    exit "$exit_code"
}

on_interrupt() {
    local elapsed
    elapsed="$(format_elapsed)"
    echo ""
    echo "[INTERRUMPIDO] $(basename "$0") detenido manualmente tras ${elapsed}." >&2
    notify_discord \
        "⚠️ Experimento interrumpido: $(basename "$0") (${HOST_TAG})" \
        "Host: \`${HOST_TAG}\`\nEl proceso fue interrumpido (SIGINT/SIGTERM) tras \`${elapsed}\`." \
        16753920
    exit 130
}

trap 'on_error $LINENO' ERR
trap 'on_interrupt' INT TERM

# Activar entorno virtual
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-bcp}}"

# Instancias en común que alcanzaron TIME_LIMIT en Gurobi y CPLEX (8 combinaciones x 2 solvers = 16 corridas)
DEFAULT_INSTANCES="er_n100_p006"

PARALLEL_MODE=0
SOLVERS="${SOLVERS:-gurobi cplex}"
K_VALUES="${K_VALUES:-3 4}"
EPSILONS="${EPSILONS:-0.00 0.05 0.10 0.15}"
TIME_LIMIT="${TIME_LIMIT:-600}"

usage() {
    cat <<EOF
Uso: $(basename "$0") [opciones]

Ejecuta las instancias que alcanzaron TIME_LIMIT en común para Gurobi y CPLEX:
  - Instancia : er_n100_p006
  - k         : {3, 4}
  - ε         : {0.00, 0.05, 0.10, 0.15}
  - Time Limit: ${TIME_LIMIT}s por corrida

Guarda el resultado en outputs/results/common_timelimit_600s_latest.csv
(sin sobrescribir main_experiments_latest.csv).

Opciones:
  --parallel        Ejecuta Gurobi y CPLEX en paralelo (8 hilos c/u) y une los CSV.
                    Por defecto se ejecutan secuencialmente (16 hilos).
  --gurobi-only     Ejecuta únicamente Gurobi.
  --cplex-only      Ejecuta únicamente CPLEX.
  --time-limit SEG  Sobrescribe el límite de tiempo por corrida (por defecto: ${TIME_LIMIT}s).
  -h, --help        Muestra esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            PARALLEL_MODE=1
            shift
            ;;
        --gurobi-only)
            SOLVERS="gurobi"
            PARALLEL_MODE=0
            shift
            ;;
        --cplex-only)
            SOLVERS="cplex"
            PARALLEL_MODE=0
            shift
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Opción no reconocida: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# shellcheck disable=SC2206
INST_ARRAY=(${INSTANCES:-$DEFAULT_INSTANCES})
INSTANCE_ARGS=(--instances "${INST_ARRAY[@]}")
INST_DESC="${INST_ARRAY[*]}"

# shellcheck disable=SC2206
K_ARRAY=($K_VALUES)
# shellcheck disable=SC2206
EPS_ARRAY=($EPSILONS)
# shellcheck disable=SC2206
SOLVER_ARRAY=($SOLVERS)

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_DIR="${PROJECT_ROOT}/outputs/results"
LOGS_DIR="${PROJECT_ROOT}/outputs/logs/common_tl${TIME_LIMIT}_${STAMP}"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
FINAL_CSV="${RESULTS_DIR}/common_timelimit_${TIME_LIMIT}s_${STAMP}.csv"
LATEST_CSV="${RESULTS_DIR}/common_timelimit_${TIME_LIMIT}s_latest.csv"

MODE_LABEL=$([[ "$PARALLEL_MODE" -eq 1 && "${#SOLVER_ARRAY[@]}" -ge 2 ]] && echo "Paralelo (8 hilos/solver)" || echo "Secuencial (16 hilos)")

echo "=============================================================================="
echo " Re-ejecución TIME_LIMIT Común BCP_k (experiments/run_experiments.py)"
echo " CPU objetivo : Intel(R) Xeon(R) Gold 5320T @ 2.30GHz (${HOST_TAG})"
echo " Modo         : ${MODE_LABEL}"
echo " Instancias   : ${INST_DESC}"
echo " Solvers      : ${SOLVER_ARRAY[*]}"
echo " Valores de k : ${K_ARRAY[*]}"
echo " Epsilons     : ${EPS_ARRAY[*]}"
echo " Time Limit   : ${TIME_LIMIT}s por corrida"
echo " CSV salida   : ${FINAL_CSV}"
echo "=============================================================================="

notify_discord \
    "🚀 Inicio: $(basename "$0") (${HOST_TAG})" \
    "Modo: \`${MODE_LABEL}\`\nSolvers: \`${SOLVER_ARRAY[*]}\`\nInstancias: \`${INST_DESC}\`\nk: \`${K_ARRAY[*]}\` | eps: \`${EPS_ARRAY[*]}\` | tl: \`${TIME_LIMIT}s\`" \
    3447003

if [[ "$PARALLEL_MODE" -eq 1 && "${#SOLVER_ARRAY[@]}" -ge 2 ]]; then
    THREADS="${THREADS:-8}"
    echo "[Modo PARALELO] Lanzando Gurobi y CPLEX en simultáneo (${THREADS} hilos físicos c/u)..."
    CSV_GRB="${RESULTS_DIR}/_part_tl${TIME_LIMIT}_gurobi_${STAMP}.csv"
    CSV_CPX="${RESULTS_DIR}/_part_tl${TIME_LIMIT}_cplex_${STAMP}.csv"

    python experiments/run_experiments.py \
        --solvers gurobi \
        "${INSTANCE_ARGS[@]}" \
        --k-values "${K_ARRAY[@]}" \
        --epsilons "${EPS_ARRAY[@]}" \
        --time-limit "$TIME_LIMIT" \
        --threads "$THREADS" \
        --output-csv "$CSV_GRB" \
        > "${LOGS_DIR}/stdout_gurobi.log" 2>&1 &
    PID_GRB=$!

    python experiments/run_experiments.py \
        --solvers cplex \
        "${INSTANCE_ARGS[@]}" \
        --k-values "${K_ARRAY[@]}" \
        --epsilons "${EPS_ARRAY[@]}" \
        --time-limit "$TIME_LIMIT" \
        --threads "$THREADS" \
        --output-csv "$CSV_CPX" \
        > "${LOGS_DIR}/stdout_cplex.log" 2>&1 &
    PID_CPX=$!

    echo "  -> Gurobi PID=${PID_GRB} (log: ${LOGS_DIR}/stdout_gurobi.log)"
    echo "  -> CPLEX  PID=${PID_CPX} (log: ${LOGS_DIR}/stdout_cplex.log)"
    echo "Esperando término de ambos procesos..."

    STATUS_GRB=0
    STATUS_CPX=0
    wait "$PID_GRB" || STATUS_GRB=$?
    wait "$PID_CPX" || STATUS_CPX=$?

    if [[ "$STATUS_GRB" -ne 0 || "$STATUS_CPX" -ne 0 ]]; then
        echo "[AVISO] Alguno de los procesos paralelos retornó código distinto de cero (Gurobi=${STATUS_GRB}, CPLEX=${STATUS_CPX})." >&2
        notify_discord \
            "⚠️ Aviso en subproceso paralelo (${HOST_TAG})" \
            "Gurobi exit code: \`${STATUS_GRB}\`\nCPLEX exit code: \`${STATUS_CPX}\`\nTiempo: \`$(format_elapsed)\`" \
            16753920
    fi

    python - "$CSV_GRB" "$CSV_CPX" "$FINAL_CSV" <<'PYEOF'
from pathlib import Path
import sys
import pandas as pd

paths = [Path(p) for p in sys.argv[1:-1] if Path(p).exists()]
out = Path(sys.argv[-1])
if not paths:
    raise SystemExit("No se generaron archivos CSV parciales.")
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df.to_csv(out, index=False, encoding="utf-8")
for p in paths:
    p.unlink()
print(f"CSV consolidado guardado en: {out} ({len(df)} filas)")
PYEOF

else
    THREADS="${THREADS:-16}"
    echo "[Modo SECUENCIAL] Ejecutando con ${THREADS} hilos físicos dedicados por corrida..."
    python experiments/run_experiments.py \
        --solvers "${SOLVER_ARRAY[@]}" \
        "${INSTANCE_ARGS[@]}" \
        --k-values "${K_ARRAY[@]}" \
        --epsilons "${EPS_ARRAY[@]}" \
        --time-limit "$TIME_LIMIT" \
        --threads "$THREADS" \
        --output-csv "$FINAL_CSV" \
        2>&1 | tee "${LOGS_DIR}/stdout_main.log"
fi

if [[ -f "$FINAL_CSV" ]]; then
    cp -f "$FINAL_CSV" "$LATEST_CSV"
fi

SUMMARY_STATS=$(python - "$FINAL_CSV" <<'PYEOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
total = len(df)
opt = int((df["status"] == "OPTIMAL").sum()) if "status" in df else 0
tl = int((df["status"] == "TIME_LIMIT").sum()) if "status" in df else 0
err = int((df["status"] == "ERROR").sum()) if "status" in df else 0
print(f"Total={total} | OPTIMAL={opt} | TIME_LIMIT={tl} | ERROR={err}")
PYEOF
)
ELAPSED_TOTAL="$(format_elapsed)"

notify_discord \
    "✅ Finalizado: $(basename "$0") (${HOST_TAG})" \
    "Modo: \`${MODE_LABEL}\`\nTiempo total: \`${ELAPSED_TOTAL}\`\nResumen: \`${SUMMARY_STATS}\`\nCSV: \`outputs/results/common_timelimit_${TIME_LIMIT}s_latest.csv\`" \
    3066993

echo ""
echo "=============================================================================="
echo " Re-ejecución finalizada en ${ELAPSED_TOTAL} (${SUMMARY_STATS})."
echo " CSV con timestamp : ${FINAL_CSV}"
echo " CSV reutilizable  : ${LATEST_CSV}"
echo "=============================================================================="
