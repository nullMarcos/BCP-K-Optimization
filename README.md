# Particionamiento Balanceado y Conexo de Grafos (BCP_k)

Proyecto desarrollado para el curso de **Optimización** de la Universidad de Concepción (UdeC). Este repositorio contiene la implementación en Python de un modelo de Programación Lineal Entera (PLE) para resolver el problema de particionamiento balanceado y conexo de grafos, comparando el rendimiento de los solvers comerciales **Gurobi** y **CPLEX** bajo restricciones avanzadas de ruptura de simetrías y tolerancia paramétrica de balance.

---
## Integrantes
- Sebastián García 
- Jesús Guevara
- Francisco Cifuentes 
- Rodrigo Bascuñán 
- Marcos Martínez 

## Estructura del Repositorio

```text
bcp_k_optimization/
├── data/
│   ├── raw/                  # Grafos originales (SNAP, Network Repository, etc.)
│   └── processed/            # Instancias preprocesadas y limpias
├── src/
│   ├── __init__.py
│   ├── model_gurobi.py       # Implementación del PLE con gurobipy (R1 - R6)
│   ├── model_cplex.py        # Implementación equivalente con docplex (CPLEX)
│   ├── callbacks.py          # Lazy constraints callback para conectividad BFS/DFS
│   ├── instance_loader.py    # Módulo de carga y generación de instancias de prueba
│   └── utils.py              # Funciones auxiliares y cálculo de métricas/gap
├── experiments/
│   ├── run_experiments.py    # Script principal para ejecución masiva por solver e instancia
│   ├── parameter_sweep.py    # Barridos automáticos del parámetro de tolerancia epsilon
│   └── generate_profiles.py  # Generación de gráficos de Perfiles de Desempeño (Dolan y Moré)
├── scripts/
│   └── test_server_connection.sh # Script de prueba y monitoreo de conexión SSH remota
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Pruebas unitarias de restricciones y factibilidad
│   ├── test_model_cplex.py   # Pruebas unitarias para CPLEX
│   └── test_callbacks.py     # Pruebas unitarias para el callback de componentes conexas
├── outputs/
│   ├── logs/                 # Archivos de registro (.log) de Gurobi y CPLEX
│   ├── results/              # Archivos CSV consolidados (tiempos, gaps, aristas cortadas)
│   └── figures/              # Gráficos y curvas trade-off generados
├── .env.example              # Plantilla para variables de entorno de conexión remota
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación y Configuración

El proyecto requiere **Python 3.9 o superior**.

```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pruebas Unitarias

Para verificar la correcta formulación del modelo y los cortes de conectividad:

```bash
pytest tests/test_model.py tests/test_callbacks.py
```

---

## Ejecución de Experimentos

### 1. Ejecución Local

Puedes ejecutar experimentos sobre una o más instancias procesadas en `data/processed/`:

```bash
# Prueba rápida con Gurobi en una instancia pequeña
python experiments/run_experiments.py \
    --solvers gurobi \
    --instances karate_club \
    --k-values 2 \
    --epsilons 0.05 \
    --time-limit 30

# Lote de instancias medianas con múltiples particiones y tolerancias
python experiments/run_experiments.py \
    --solvers gurobi \
    --instances fem_can_24 karate_club power_bcspwr01 les_miserables \
    --k-values 2 3 \
    --epsilons 0.00 0.05 0.10 \
    --time-limit 60
```

Los resultados se guardan automáticamente en:
- Métricas consolidadas: `outputs/<TIMESTAMP>/experiment_results.csv`
- Registros detallados: `outputs/<TIMESTAMP>/logs/<SOLVER>/*.log`

---

### 2. Ejecución Remota en Servidor (Chome / Luthier)

Para ejecutar en el servidor remoto de cómputo:

#### Configuración de variables (opcional para notificaciones Discord):
```bash
cp .env.example .env
# Edita .env con tus credenciales o webhook si deseas alertas en Discord
```

#### Script de verificación rápida remota:
El script `scripts/test_server_connection.sh` verifica la conexión SSH, comprueba el entorno virtual remoto y corre un experimento de prueba rápido:

```bash
chmod +x scripts/test_server_connection.sh
./scripts/test_server_connection.sh
```

#### Sincronización de resultados desde el servidor a tu máquina local:
```bash
rsync -avzP grupo12@chome.inf.udec.cl:~/BCP-K-Optimization/outputs/ ./outputs/
```

