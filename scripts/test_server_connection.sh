#!/usr/bin/env bash
set -euo pipefail

# Cargar variables locales desde .env si existe
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ==============================================================================
# 1. Variables de configuración (sobrescribibles por .env o entorno)
# ==============================================================================
REMOTE_USER="${REMOTE_USER:-grupo12}"
REMOTE_HOST="${REMOTE_HOST:-chome.inf.udec.cl}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-~/BCP-K-Optimization}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-run_bcp_gurobi.py}"
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS=(
    -p "${REMOTE_PORT}"
    -o ConnectTimeout=10
    -o BatchMode=yes
)

# ==============================================================================
# Funciones auxiliares
# ==============================================================================
notify_discord() {
    local title="$1"
    local description="$2"
    local color="$3"  # Decimal: verde=3066993, rojo=15158332, azul=3447003

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
' "$title" "$description" "$color")

    curl -fsSL -X POST -H "Content-Type: application/json" \
        -d "$payload" \
        "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || true
}

on_error() {
    local exit_code="$?"
    local line_no="${1:-unknown}"
    echo ""
    echo "[ERROR] La prueba falló en la línea ${line_no} (código de salida: ${exit_code})."
    notify_discord \
        "❌ Fallo en prueba SSH y Gurobi (${REMOTE_HOST})" \
        "Usuario: \`${REMOTE_USER}\`\nHost: \`${REMOTE_HOST}:${REMOTE_PORT}\`\nLínea: \`${line_no}\` | Exit code: \`${exit_code}\`" \
        15158332
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

# ==============================================================================
# 2. Flujo de ejecución
# ==============================================================================
echo "=============================================================================="
echo " Iniciando prueba de conexión SSH y experimento BCP (Gurobi)"
echo " Destino : ${SSH_TARGET}:${REMOTE_PORT}"
echo " Directorio remoto: ${REMOTE_DIR}"
echo "=============================================================================="

echo "[1/3] Verificando conectividad SSH básica con ${SSH_TARGET}..."
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "echo 'Conexión SSH establecida correctamente.'"

echo "[2/3] Ejecutando comprobaciones remotas (sistema, entorno y solvers)..."
REMOTE_OUTPUT=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "bash -s -- \"${REMOTE_DIR}\" \"${VENV_ACTIVATE}\" \"${EXPERIMENT_SCRIPT}\"" <<'EOF'
set -euo pipefail

RAW_DIR="$1"
VENV_PATH="$2"
EXP_SCRIPT="$3"

# Resolver directorio remoto de forma robusta
# Si empieza con ~/ quitar la tilde para anteponer $HOME
if [[ "$RAW_DIR" == "~/"* ]]; then
    TARGET_DIR="$HOME/${RAW_DIR#\~/}"
elif [[ "$RAW_DIR" == "~" ]]; then
    TARGET_DIR="$HOME"
elif [[ "$RAW_DIR" == /* ]]; then
    TARGET_DIR="$RAW_DIR"
else
    TARGET_DIR="$HOME/$RAW_DIR"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "ERROR: El directorio del proyecto no existe en el servidor: $TARGET_DIR" >&2
    echo "Sugerencia: Clona el repositorio primero con: git clone https://github.com/nullMarcos/BCP-K-Optimization.git" >&2
    exit 1
fi

cd "$TARGET_DIR"

echo "--- Directorio de trabajo en servidor ---"
pwd

echo "--- Sistema y fecha ---"
uname -a
date

echo "--- Activando entorno virtual y verificando solvers ---"
if [[ ! -f "${VENV_PATH}" ]]; then
    echo "ERROR: No se encontró el entorno virtual en $TARGET_DIR/${VENV_PATH}" >&2
    echo "Sugerencia: Crea el entorno con: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${VENV_PATH}"

echo "Python activo: $(python -V) en $(which python)"

MISSING_PKGS=()
for pkg in gurobipy docplex networkx pandas numpy; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        MISSING_PKGS+=("${pkg}")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "ERROR: Faltan paquetes en el entorno virtual del servidor: ${MISSING_PKGS[*]}" >&2
    echo "Sugerencia: Instálalos en el servidor ejecutando:" >&2
    echo "    source ${VENV_PATH} && pip install --upgrade pip && pip install ${MISSING_PKGS[*]}" >&2
    exit 1
fi

echo "Dependencias OK (gurobipy, docplex, networkx, pandas y numpy disponibles)."

echo "[3/3] Ejecutando experimento corto en karate_club (k=2, eps=0.05, time-limit=15s)..."
if [[ -f "${EXP_SCRIPT}" ]]; then
    python "${EXP_SCRIPT}" --instance karate_club -k 2 --eps 0.05 --time-limit 15
elif [[ -f "experiments/run_experiments.py" ]]; then
    python experiments/run_experiments.py --solvers gurobi --instances karate_club --k-values 2 --epsilons 0.05 --time-limit 15
else
    python - <<'PYEOF'
import time
import gurobipy as gp
import networkx as nx

t0 = time.time()
G = nx.karate_club_graph()
n = G.number_of_nodes()
k = 2
eps = 0.05
time_limit = 15

# Cota de partición balanceada simple para prueba rápida
lb = int((n / k) * (1.0 - eps))
ub = int((n / k) * (1.0 + eps)) + 1

with gp.Env(empty=True) as env:
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", time_limit)
    env.start()
    with gp.Model("bcp_karate_smoke_test", env=env) as m:
        x = m.addVars(G.nodes(), range(k), vtype=gp.GRB.BINARY, name="x")
        y = m.addVars(G.edges(), vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")

        # Cada nodo en exactamente un cluster
        for i in G.nodes():
            m.addConstr(gp.quicksum(x[i, p] for p in range(k)) == 1)

        # Cardinalidad balanceada por cluster
        for p in range(k):
            cnt = gp.quicksum(x[i, p] for i in G.nodes())
            m.addConstr(cnt >= lb)
            m.addConstr(cnt <= ub)

        # Aristas cortadas si endpoints están en clusters distintos
        for u, v in G.edges():
            for p in range(k):
                m.addConstr(y[u, v] >= x[u, p] - x[v, p])
                m.addConstr(y[u, v] >= x[v, p] - x[u, p])

        m.setObjective(gp.quicksum(y[u, v] for u, v in G.edges()), gp.GRB.MINIMIZE)
        m.optimize()

        elapsed = time.time() - t0
        print(
            f"KarateClub OK | status={m.Status} | obj={m.ObjVal:.4f} | "
            f"bounds=[{lb}, {ub}] | elapsed={elapsed:.2f}s"
        )
PYEOF
fi
EOF
)

echo "$REMOTE_OUTPUT"

SUMMARY_LINE=$(printf "%s\n" "$REMOTE_OUTPUT" | grep -E "OPTIMAL|Z\*=|Dependencias OK|KarateClub OK|Solvers OK" | tail -n 2 | tr '\n' ' ' || true)
if [[ -z "$SUMMARY_LINE" ]]; then
    SUMMARY_LINE="Ejecución finalizada con éxito."
fi

notify_discord \
    "✅ Prueba SSH y Gurobi exitosa (${REMOTE_HOST})" \
    "Usuario: \`${REMOTE_USER}\`\nHost: \`${REMOTE_HOST}:${REMOTE_PORT}\`\nInstancia: \`karate_club\` (k=2, eps=0.05, tl=15s)\nResumen: \`${SUMMARY_LINE}\`" \
    3066993

echo ""
echo "=============================================================================="
echo " PRUEBA EXITOSA: Conexión, solvers y experimento completados sin errores."
echo "=============================================================================="
