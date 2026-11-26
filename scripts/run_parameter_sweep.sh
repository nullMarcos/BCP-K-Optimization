#!/usr/bin/env bash
# ==============================================================================
# Script de barridos paramétricos para `experiments/parameter_sweep.py`
# Optimizado para Intel(R) Xeon(R) Gold 5320T CPU @ 2.30GHz (20 núcleos / 40 hilos)
#
# Subcomandos soportados:
#   ./scripts/run_parameter_sweep.sh epsilon       # Sensibilidad al balance ε
#   ./scripts/run_parameter_sweep.sh scalability   # Escalabilidad, quiebres y gap vs t
#   ./scripts/run_parameter_sweep.sh ablation_r6   # Ablación Con R6 vs Sin R6
#   ./scripts/run_parameter_sweep.sh all           # Ejecuta los 3 barridos en orden
#
# Reutilización del CSV maestro (sin volver a resolver):
#   ./scripts/run_parameter_sweep.sh epsilon --from-csv
#   ./scripts/run_parameter_sweep.sh scalability --from-csv
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
    echo "[ERROR] $(basename "$0") (${MODE:-unknown}) falló en la línea ${line_no} (código: ${exit_code}, tiempo: ${elapsed})." >&2
    notify_discord \
        "❌ Fallo en run_parameter_sweep.sh [${MODE:-unknown}] (${HOST_TAG})" \
        "Host: \`${HOST_TAG}\`\nSubcomando: \`${MODE:-unknown}\`\nLínea: \`${line_no}\` | Exit code: \`${exit_code}\`\nTiempo transcurrido: \`${elapsed}\`" \
        15158332
    exit "$exit_code"
}

on_interrupt() {
    local elapsed
    elapsed="$(format_elapsed)"
    echo ""
    echo "[INTERRUMPIDO] $(basename "$0") (${MODE:-unknown}) detenido manualmente tras ${elapsed}." >&2
    notify_discord \
        "⚠️ Barrido interrumpido: run_parameter_sweep.sh [${MODE:-unknown}] (${HOST_TAG})" \
        "Host: \`${HOST_TAG}\`\nSubcomando: \`${MODE:-unknown}\` detenido (SIGINT/SIGTERM) tras \`${elapsed}\`." \
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

EPSILON_INSTANCES="fem_can_24 karate_club fem_can_61 les_miserables grid_10x10 ba_n100_m2 power_bcspwr03"
SCALABILITY_INSTANCES="fem_can_24 karate_club fem_can_61 les_miserables grid_10x10 ba_n100_m2 er_n100_p006 power_bcspwr03 ba_n150_m2 snap_facebook"
ABLATION_INSTANCES="fem_can_24 karate_club fem_can_61 les_miserables power_bcspwr03"
DEFAULT_LATEST_CSV="${PROJECT_ROOT}/outputs/results/main_experiments_latest.csv"

MODE="${1:-all}"
if [[ "$MODE" =~ ^(-h|--help)$ ]]; then
    MODE="help"
else
    shift || true
fi

PARALLEL_MODE=0
USE_ALL_INSTANCES=0
SOLVERS="${SOLVERS:-gurobi cplex}"
CUSTOM_TIME_LIMIT=""
FROM_CSV=""

usage() {
    cat <<EOF
Uso: $(basename "$0") {epsilon|scalability|ablation_r6|all} [opciones]

Subcomandos:
  epsilon         Barrido de tolerancia ε ∈ {0.00, 0.05, 0.10, 0.15} con k ∈ {2, 3}.
  scalability     Barrido de escalabilidad con k ∈ {2, 3, 4} y ε = 0.05.
  ablation_r6     Estudio de ablación Con R6 vs Sin R6 con k ∈ {2, 3, 4}.
  all             Ejecuta epsilon, scalability y ablation_r6 secuencialmente.

Opciones:
  --from-csv [CSV]  No ejecuta los solvers: analiza un CSV existente (si no se indica
                    ruta, usa outputs/results/main_experiments_latest.csv).
  --parallel        Ejecuta Gurobi y CPLEX en paralelo (8 hilos c/u).
                    Por defecto corre secuencialmente con 16 hilos dedicados.
  --all-instances   Usa las 18 instancias de data/processed/.
  --gurobi-only     Ejecuta únicamente Gurobi.
  --cplex-only      Ejecuta únicamente CPLEX.
  --time-limit SEG  Sobrescribe el límite de tiempo por corrida.
  -h, --help        Muestra esta ayuda.
EOF
}

if [[ "$MODE" == "help" ]]; then
    usage
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-csv)
            if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
                FROM_CSV="$2"
                shift 2
            else
                FROM_CSV="$DEFAULT_LATEST_CSV"
                shift
            fi
            ;;
        --parallel)
            PARALLEL_MODE=1
            shift
            ;;
        --all-instances)
            USE_ALL_INSTANCES=1
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
            CUSTOM_TIME_LIMIT="$2"
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
SOLVER_ARRAY=($SOLVERS)

SOURCE_DESC=$([[ -n "$FROM_CSV" ]] && echo "Desde CSV (${FROM_CSV##*/})" || echo "Ejecución en solvers (${SOLVER_ARRAY[*]})")
notify_discord \
    "🚀 Inicio: run_parameter_sweep.sh [${MODE}] (${HOST_TAG})" \
    "Subcomando: \`${MODE}\`\nModo: \`${SOURCE_DESC}\`" \
    3447003

run_single_sweep() {
    local sweep_name="$1"
    local default_inst="$2"
    local k_vals="$3"
    local eps_vals="$4"
    local default_tl="$5"

    local tl="${CUSTOM_TIME_LIMIT:-$default_tl}"

    if [[ -n "$FROM_CSV" && "$sweep_name" != "ablation_r6" ]]; then
        if [[ ! -f "$FROM_CSV" ]]; then
            echo "ERROR: No existe el archivo CSV indicado: $FROM_CSV" >&2
            exit 1
        fi
        echo ""
        echo "------------------------------------------------------------------------------"
        echo " Analizando barrido '${sweep_name}' desde CSV existente: ${FROM_CSV}"
        echo "------------------------------------------------------------------------------"
        local target_csv="$FROM_CSV"
        if [[ "$sweep_name" == "scalability" ]]; then
            local filtered_csv="${PROJECT_ROOT}/outputs/results/_filtered_scalability_eps005.csv"
            python - "$FROM_CSV" "$filtered_csv" <<'PYEOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
if "epsilon" in df.columns and df["epsilon"].nunique() > 1:
    sub = df[df["epsilon"].round(4) == 0.05]
    if len(sub) > 0:
        df = sub
df.to_csv(sys.argv[2], index=False, encoding="utf-8")
PYEOF
            target_csv="$filtered_csv"
        fi
        python experiments/parameter_sweep.py "$sweep_name" \
            --from-csv "$target_csv" \
            --time-limit "$tl"
        if [[ "$target_csv" != "$FROM_CSV" && -f "$target_csv" ]]; then
            rm -f "$target_csv"
        fi
        return 0
    fi

    local inst_args=()
    if [[ "$USE_ALL_INSTANCES" -eq 0 ]]; then
        # shellcheck disable=SC2206
        local inst_arr=($default_inst)
        inst_args=(--instances "${inst_arr[@]}")
    fi
    # shellcheck disable=SC2206
    local k_arr=($k_vals)
    # shellcheck disable=SC2206
    local eps_arr=($eps_vals)

    echo ""
    echo "------------------------------------------------------------------------------"
    echo " Ejecutando barrido: ${sweep_name}"
    echo " Solvers: ${SOLVER_ARRAY[*]} | k: ${k_arr[*]} | eps: ${eps_arr[*]} | tl: ${tl}s"
    echo "------------------------------------------------------------------------------"

    if [[ "$PARALLEL_MODE" -eq 1 && "${#SOLVER_ARRAY[@]}" -ge 2 && "$sweep_name" != "ablation_r6" ]]; then
        local threads="${THREADS:-8}"
        local stamp
        stamp="$(date +%Y-%m-%d_%H-%M-%S)"
        local res_dir="${PROJECT_ROOT}/outputs/results"
        local log_dir="${PROJECT_ROOT}/outputs/logs/sweep_${sweep_name}_${stamp}"
        mkdir -p "$res_dir" "$log_dir"

        local csv_grb="${res_dir}/_part_${sweep_name}_gurobi_${stamp}.csv"
        local csv_cpx="${res_dir}/_part_${sweep_name}_cplex_${stamp}.csv"
        local csv_merged="${res_dir}/sweep_${sweep_name}_${stamp}.csv"

        echo "[Modo PARALELO] Corriendo Gurobi y CPLEX en simultáneo (${threads} hilos c/u)..."
        python experiments/run_experiments.py \
            --solvers gurobi \
            "${inst_args[@]}" \
            --k-values "${k_arr[@]}" \
            --epsilons "${eps_arr[@]}" \
            --time-limit "$tl" \
            --threads "$threads" \
            --output-csv "$csv_grb" \
            > "${log_dir}/stdout_gurobi.log" 2>&1 &
        local pid_grb=$!

        python experiments/run_experiments.py \
            --solvers cplex \
            "${inst_args[@]}" \
            --k-values "${k_arr[@]}" \
            --epsilons "${eps_arr[@]}" \
            --time-limit "$tl" \
            --threads "$threads" \
            --output-csv "$csv_cpx" \
            > "${log_dir}/stdout_cplex.log" 2>&1 &
        local pid_cpx=$!

        local st_grb=0
        local st_cpx=0
        wait "$pid_grb" || st_grb=$?
        wait "$pid_cpx" || st_cpx=$?

        if [[ "$st_grb" -ne 0 || "$st_cpx" -ne 0 ]]; then
            notify_discord \
                "⚠️ Aviso en subproceso paralelo de ${sweep_name} (${HOST_TAG})" \
                "Gurobi exit: \`${st_grb}\` | CPLEX exit: \`${st_cpx}\`" \
                16753920
        fi

        python - "$csv_grb" "$csv_cpx" "$csv_merged" <<'PYEOF'
from pathlib import Path
import sys
import pandas as pd

paths = [Path(p) for p in sys.argv[1:-1] if Path(p).exists()]
out = Path(sys.argv[-1])
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df.to_csv(out, index=False, encoding="utf-8")
for p in paths:
    p.unlink()
print(f"CSV consolidado de barrido: {out} ({len(df)} filas)")
PYEOF

        python experiments/parameter_sweep.py "$sweep_name" \
            --from-csv "$csv_merged" \
            --time-limit "$tl"
    else
        local threads="${THREADS:-16}"
        echo "[Modo SECUENCIAL] Ejecutando con ${threads} hilos físicos dedicados..."
        python experiments/parameter_sweep.py "$sweep_name" \
            --solvers "${SOLVER_ARRAY[@]}" \
            "${inst_args[@]}" \
            --k-values "${k_arr[@]}" \
            --epsilons "${eps_arr[@]}" \
            --time-limit "$tl" \
            --threads "$threads"
    fi
}

case "$MODE" in
    epsilon)
        run_single_sweep "epsilon" "$EPSILON_INSTANCES" "${K_VALUES:-2 3}" "${EPSILONS:-0.00 0.05 0.10 0.15}" "90"
        ;;
    scalability)
        run_single_sweep "scalability" "$SCALABILITY_INSTANCES" "${K_VALUES:-2 3 4}" "${EPSILONS:-0.05}" "90"
        ;;
    ablation_r6)
        run_single_sweep "ablation_r6" "$ABLATION_INSTANCES" "${K_VALUES:-2 3 4}" "${EPSILONS:-0.05}" "60"
        ;;
    all)
        run_single_sweep "epsilon" "$EPSILON_INSTANCES" "2 3" "0.00 0.05 0.10 0.15" "90"
        run_single_sweep "scalability" "$SCALABILITY_INSTANCES" "2 3 4" "0.05" "90"
        run_single_sweep "ablation_r6" "$ABLATION_INSTANCES" "2 3 4" "0.05" "60"
        ;;
    *)
        echo "Subcomando no válido: $MODE" >&2
        usage
        exit 1
        ;;
esac

ELAPSED_TOTAL="$(format_elapsed)"
notify_discord \
    "✅ Finalizado: run_parameter_sweep.sh [${MODE}] (${HOST_TAG})" \
    "Subcomando: \`${MODE}\`\nModo: \`${SOURCE_DESC}\`\nTiempo total: \`${ELAPSED_TOTAL}\`\nSalidas actualizadas en \`outputs/results/\` y \`outputs/figures/\`." \
    3066993

echo ""
echo "=============================================================================="
echo " Barrido(s) [${MODE}] finalizado(s) con éxito en ${ELAPSED_TOTAL}."
echo " Tablas en : outputs/results/"
echo " Figuras en: outputs/figures/"
echo "=============================================================================="
