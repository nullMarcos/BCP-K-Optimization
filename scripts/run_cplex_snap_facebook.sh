#!/usr/bin/env bash
# ==============================================================================
# Script de ejecución dedicado para CPLEX sobre `snap_facebook` (k=4, ε=0.05),
# con límite de tiempo de 1800 segundos (30 minutos) y sin paralelismo entre solvers.
#
# Guarda su propio CSV en `outputs/results/cplex_snap_facebook_k4_eps005_1800s_*.csv`
# sin sobrescribir `main_experiments_latest.csv`.
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

format_elapsed() {
    local secs=$(( $(date +%s) - START_EPOCH ))
    printf "%02dh:%02dm:%02ds" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

notify_discord() {
    local title="$1"
    local description="$2"
    local color="$3"

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

VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-bcp}}"

INSTANCE="snap_facebook"
SOLVER="cplex"
K_VAL="4"
EPS_VAL="0.05"
TIME_LIMIT="${TIME_LIMIT:-1800}"
THREADS="${THREADS:-16}"

usage() {
    cat <<EOF
Uso: $(basename "$0") [opciones]

Ejecuta únicamente CPLEX sobre:
  - Instancia : ${INSTANCE}
  - k         : ${K_VAL}
  - ε         : ${EPS_VAL}
  - Hilos     : ${THREADS}
  - Time Limit: ${TIME_LIMIT}s (30 min)

Guarda el resultado en outputs/results/cplex_snap_facebook_k4_eps005_${TIME_LIMIT}s_latest.csv.

Opciones:
  --time-limit SEG  Sobrescribe el límite de tiempo (por defecto: ${TIME_LIMIT}s).
  --threads N       Sobrescribe el número de hilos (por defecto: ${THREADS}).
  -h, --help        Muestra esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
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

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_DIR="${PROJECT_ROOT}/outputs/results"
LOGS_DIR="${PROJECT_ROOT}/outputs/logs/cplex_snap_facebook_tl${TIME_LIMIT}_${STAMP}"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
FINAL_CSV="${RESULTS_DIR}/cplex_snap_facebook_k4_eps005_${TIME_LIMIT}s_${STAMP}.csv"
LATEST_CSV="${RESULTS_DIR}/cplex_snap_facebook_k4_eps005_${TIME_LIMIT}s_latest.csv"

echo "=============================================================================="
echo " Ejecución Dedicada BCP_k: CPLEX en ${INSTANCE}"
echo " Host         : ${HOST_TAG}"
echo " Solver       : CPLEX (${THREADS} hilos)"
echo " Instancia    : ${INSTANCE} | k=${K_VAL} | ε=${EPS_VAL}"
echo " Time Limit   : ${TIME_LIMIT}s (30 min)"
echo " CSV salida   : ${FINAL_CSV}"
echo "=============================================================================="

notify_discord \
    "🚀 Inicio: $(basename "$0") (${HOST_TAG})" \
    "Solver: \`CPLEX\` (\`${THREADS}\` hilos)\nInstancia: \`${INSTANCE}\` | k=\`${K_VAL}\` | eps=\`${EPS_VAL}\` | tl=\`${TIME_LIMIT}s\`" \
    3447003

python experiments/run_experiments.py \
    --solvers "$SOLVER" \
    --instances "$INSTANCE" \
    --k-values "$K_VAL" \
    --epsilons "$EPS_VAL" \
    --time-limit "$TIME_LIMIT" \
    --threads "$THREADS" \
    --output-csv "$FINAL_CSV" \
    2>&1 | tee "${LOGS_DIR}/stdout_cplex.log"

if [[ -f "$FINAL_CSV" ]]; then
    cp -f "$FINAL_CSV" "$LATEST_CSV"
fi

SUMMARY_STATS=$(python - "$FINAL_CSV" <<'PYEOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
if df.empty:
    print("Sin filas")
else:
    r = df.iloc[0]
    gap = f"{float(r['mip_gap'])*100:.2f}%" if pd.notna(r.get("mip_gap")) else "N/A"
    print(f"status={r['status']} | obj={r['objective']} | bound={r['best_bound']} | gap={gap} | t={r['runtime']}s")
PYEOF
)
ELAPSED_TOTAL="$(format_elapsed)"

notify_discord \
    "✅ Finalizado: $(basename "$0") (${HOST_TAG})" \
    "Tiempo total: \`${ELAPSED_TOTAL}\`\nResultado: \`${SUMMARY_STATS}\`\nCSV: \`outputs/results/cplex_snap_facebook_k4_eps005_${TIME_LIMIT}s_latest.csv\`" \
    3066993

echo ""
echo "=============================================================================="
echo " Finalizado en ${ELAPSED_TOTAL} (${SUMMARY_STATS})."
echo " CSV con timestamp : ${FINAL_CSV}"
echo " CSV reutilizable  : ${LATEST_CSV}"
echo "=============================================================================="
