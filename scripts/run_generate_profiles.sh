#!/usr/bin/env bash
# ==============================================================================
# Script para generar Perfiles de Desempeño de Dolan & Moré, boxplots de tiempo,
# diagramas de brecha de optimalidad y actividad de Branch-and-Cut
# (`experiments/generate_profiles.py`), más figuras demostrativas de particiones.
#
# Incluye notificaciones automáticas a Discord vía DISCORD_WEBHOOK_URL (en .env).
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
        "❌ Fallo en run_generate_profiles.sh (${HOST_TAG})" \
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
        "⚠️ Generación de perfiles interrumpida (${HOST_TAG})" \
        "Host: \`${HOST_TAG}\`\nDetenido manualmente tras \`${elapsed}\`." \
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

TIME_LIMIT="${TIME_LIMIT:-90}"
WITH_PARTITIONS=0
CSV_FILES=()

usage() {
    cat <<EOF
Uso: $(basename "$0") [archivos.csv ...] [opciones]

Si no se indican archivos CSV, busca automáticamente los resultados generados en:
  - outputs/results/main_experiments_latest.csv
  - outputs/results/main_experiments_*.csv
  - outputs/results/sweep_*.csv
  - outputs/*/experiment_results.csv

Opciones:
  --time-limit SEG     Límite de tiempo de referencia para las líneas punteadas (por defecto: ${TIME_LIMIT}s).
  --with-partitions    Genera además figuras demostrativas de particiones (Con/Sin R4 y ε) en outputs/figures/.
  -h, --help           Muestra esta ayuda.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --with-partitions)
            WITH_PARTITIONS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            CSV_FILES+=("$1")
            shift
            ;;
    esac
done

if [[ "${#CSV_FILES[@]}" -eq 0 ]]; then
    if [[ -f "outputs/results/main_experiments_latest.csv" ]]; then
        CSV_FILES=("outputs/results/main_experiments_latest.csv")
    else
        shopt -s nullglob
        CSV_FILES=(
            outputs/results/main_experiments_*.csv
            outputs/results/sweep_epsilon_*.csv
            outputs/results/sweep_scalability_*.csv
        )
        if [[ "${#CSV_FILES[@]}" -eq 0 ]]; then
            CSV_FILES=(outputs/*/experiment_results.csv)
        fi
        shopt -u nullglob
    fi
fi

if [[ "${#CSV_FILES[@]}" -eq 0 ]]; then
    echo "ERROR: No se encontraron archivos CSV de resultados en outputs/." >&2
    echo "Ejecuta primero ./scripts/run_main_experiments.sh o ./scripts/run_parameter_sweep.sh" >&2
    exit 1
fi

echo "=============================================================================="
echo " Generando Perfiles de Dolan & Moré y Análisis de Branch-and-Cut"
echo " Archivos CSV (${#CSV_FILES[@]}):"
for f in "${CSV_FILES[@]}"; do
    echo "   - $f"
done
echo " Time limit de referencia: ${TIME_LIMIT}s"
echo "=============================================================================="

python experiments/generate_profiles.py "${CSV_FILES[@]}" --time-limit "$TIME_LIMIT"

if [[ "$WITH_PARTITIONS" -eq 1 ]]; then
    echo ""
    echo "------------------------------------------------------------------------------"
    echo " Generando figuras demostrativas de particiones (karate_club y grid_10x10)..."
    echo "------------------------------------------------------------------------------"
    python scripts/visualize_partition.py --instance karate_club --k 2 --epsilon 0.05 --compare-r4
    python scripts/visualize_partition.py --instance karate_club --k 2 --compare-epsilons 0.00 0.15
    python scripts/visualize_partition.py --instance grid_10x10 --k 2 --epsilon 0.05
fi

ELAPSED_TOTAL="$(format_elapsed)"
notify_discord \
    "✅ Finalizado: run_generate_profiles.sh (${HOST_TAG})" \
    "Archivos CSV procesados: \`${#CSV_FILES[@]}\`\nFiguras de particiones: \`$([[ "$WITH_PARTITIONS" -eq 1 ]] && echo 'Sí' || echo 'No')\`\nTiempo: \`${ELAPSED_TOTAL}\`" \
    3066993

echo ""
echo "=============================================================================="
echo " Gráficos y resúmenes generados en ${ELAPSED_TOTAL}:"
echo "   - outputs/figures/dolan_more_runtime.png"
echo "   - outputs/figures/boxplot_runtime.png"
echo "   - outputs/figures/gap_diagram.png"
echo "   - outputs/figures/branch_and_cut_activity.png"
echo "   - outputs/results/performance_ratios.csv"
echo "   - outputs/results/branch_and_cut_summary.csv"
echo "=============================================================================="
