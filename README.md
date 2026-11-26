# Particionamiento Balanceado y Conexo de Grafos (BCP_k)

Proyecto desarrollado para el curso de **Optimización** de la Universidad de Concepción (UdeC). Este repositorio contiene la implementación en Python de un modelo de Programación Lineal Entera (PLE) mediante **Branch-and-Cut** para resolver el problema de particionamiento balanceado y conexo de grafos ($BCP_k$), comparando el rendimiento de los solvers comerciales **Gurobi** y **CPLEX** bajo restricciones avanzadas de ruptura de simetrías (**R6**), cortes perezosos de conectividad por separadores nodales minimales (**R4**) y tolerancia paramétrica de balance (**R2**).

---

## Integrantes
- Sebastián García
- Jesús Guevara
- Francisco Cifuentes
- Rodrigo Bascuñán
- Marcos Martínez

---

## Estructura del Repositorio

```text
bcp_k_optimization/
├── data/
│   ├── raw/                        # Grafos originales comprimidos (SNAP, Matrix Market)
│   ├── processed/                  # 18 instancias preprocesadas en formato .adjlist
│   └── graphml/                    # Grafos exportados a .graphml para Gephi / Cytoscape / yEd
├── src/
│   ├── __init__.py                 # Exportación perezosa (lazy imports) del paquete
│   ├── model_gurobi.py             # Implementación del PLE con gurobipy (R1 - R6)
│   ├── model_cplex.py              # Implementación equivalente con docplex / CPLEX
│   ├── callbacks.py                # Separador BFS/DFS, poda minimal y Lazy Callback (R4)
│   ├── instance_loader.py          # Módulo de carga y generación de instancias de prueba
│   ├── license.py                  # Detección y validación multiplataforma de gurobi.lic
│   └── utils.py                    # Funciones compartidas, cotas R2 y despachador solve_bcp
├── experiments/
│   ├── run_experiments.py          # Motor principal de ejecución por solver, instancia, k y ε
│   ├── parameter_sweep.py          # Barridos de sensibilidad (ε), escalabilidad y ablación R6
│   └── generate_profiles.py        # Perfiles de Dolan y Moré, boxplots y actividad Branch-and-Cut
├── scripts/
│   ├── run_main_experiments.sh     # Script maestro (10 instancias curadas, paralelo o secuencial)
│   ├── run_parameter_sweep.sh      # Script para barridos (epsilon, scalability, ablation_r6)
│   ├── run_generate_profiles.sh    # Script para generar gráficos y perfiles desde CSV
│   ├── export_graphml.py           # Convierte los .adjlist de data/processed/ a .graphml
│   ├── visualize_partition.py      # Dibuja particiones (PNG) y exporta .graphml coloreados
│   └── test_server_connection.sh   # Verificación de conexión SSH y entorno remoto
├── tests/
│   ├── __init__.py
│   ├── test_model.py               # Pruebas unitarias de restricciones y factibilidad (Gurobi)
│   ├── test_model_cplex.py         # Pruebas unitarias y validación cruzada (CPLEX)
│   └── test_callbacks.py           # Pruebas unitarias y de estrés para el separador nodal R4
├── outputs/
│   ├── logs/                       # Archivos de registro (.log) nativos de Gurobi y CPLEX
│   ├── results/                    # Tablas CSV consolidadas (tiempos, gaps, quiebres, ablación)
│   └── figures/                    # Gráficos y curvas generadas en alta resolución (300 DPI)
├── .env.example                    # Plantilla para variables de entorno de conexión remota
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación y Configuración

El proyecto requiere **Python 3.9+**.

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

Para verificar la formulación del modelo, cotas de balance y cortes de conectividad:

```bash
pytest tests/test_model.py tests/test_callbacks.py
```

---

## Flujo Recomendado de Ejecución de Experimentos

Los scripts en `scripts/` están sintonizados para el procesador **`Intel(R) Xeon(R) Gold 5320T CPU @ 2.30GHz`** (20 núcleos físicos / 40 hilos), evaluando por defecto la selección curada de **10 instancias** representativas de las tres familias topológicas (`fem_can_24`, `karate_club`, `fem_can_61`, `les_miserables`, `grid_10x10`, `ba_n100_m2`, `er_n100_p006`, `power_bcspwr03`, `ba_n150_m2`, `snap_facebook`).

### Estrategia rápida en 4 pasos (reutilizando el CSV maestro)

#### 1. Ejecutar el experimento maestro en paralelo (~40 – 45 min)
Corre Gurobi (8 hilos físicos) y CPLEX (8 hilos físicos) en simultáneo para todas las combinaciones de $k \in \{2, 3, 4\}$ y $\varepsilon \in \{0.00, 0.05, 0.10, 0.15\}$ con límite de 90 s por corrida, y guarda el resultado consolidado en `outputs/results/main_experiments_latest.csv`:

```bash
./scripts/run_main_experiments.sh --parallel
```
*(Si prefieres ejecutar un solver a la vez con 16 hilos físicos dedicados, omite `--parallel`: `./scripts/run_main_experiments.sh`).*

#### 2. Generar análisis de sensibilidad ($\varepsilon$) y escalabilidad desde el CSV maestro (~5 s)
Reutiliza `outputs/results/main_experiments_latest.csv` mediante `--from-csv` sin volver a resolver las instancias:

```bash
# Curva y tabla de compromiso de balance (outputs/figures/epsilon_tradeoff.png)
./scripts/run_parameter_sweep.sh epsilon --from-csv

# Escalabilidad, puntos de quiebre (global y por familia) y MIPGap vs tiempo
./scripts/run_parameter_sweep.sh scalability --from-csv
```

#### 3. Ejecutar de forma individual el estudio de ablación de Ruptura de Simetrías R6 (~12 – 15 min)
Compara el desempeño **Con R6 vs. Sin R6** ($k \in \{2, 3, 4\}$) sobre las instancias pequeñas y medianas:

```bash
./scripts/run_parameter_sweep.sh ablation_r6 --gurobi-only
```

#### 4. Generar Perfiles de Dolan & Moré, actividad de Branch-and-Cut y particiones (~10 s)
Genera `dolan_more_runtime.png`, `boxplot_runtime.png`, `gap_diagram.png`, `branch_and_cut_activity.png` y las figuras comparativas de grafos particionados:

```bash
./scripts/run_generate_profiles.sh --with-partitions --time-limit 90
```

---

## Ejecución Manual con Scripts de Python y Visualización

### 1. Corridas personalizadas con `run_experiments.py`
```bash
# Prueba rápida con Gurobi en karate_club fijando 4 hilos
python experiments/run_experiments.py \
    --solvers gurobi \
    --instances karate_club \
    --k-values 2 3 \
    --epsilons 0.05 \
    --threads 4 \
    --time-limit 30
```

### 2. Exportación a GraphML y visualización de particiones
Para convertir los 18 grafos de `data/processed/` a `.graphml` (compatibles con **Gephi**, **Cytoscape** y **yEd**):
```bash
python scripts/export_graphml.py
```

Para resolver una instancia, dibujar su partición en `.png` y exportar el `.graphml` con atributos de bloque (`partition`, `block_size`) y aristas cortadas (`is_cut_edge`):
```bash
# Comparar visualmente Sin R4 (desconectado) vs Con R4 (conexo)
python scripts/visualize_partition.py --instance karate_club --k 2 --epsilon 0.05 --compare-r4

# Comparar el efecto de relajar el balance (eps=0.00 vs eps=0.15)
python scripts/visualize_partition.py --instance karate_club --k 2 --compare-epsilons 0.00 0.15
```

---

## Ejecución Remota en Servidor (Chome / Luthier)

1. **Verificación rápida de conexión SSH y entorno remoto:**
   ```bash
   chmod +x scripts/test_server_connection.sh
   ./scripts/test_server_connection.sh
   ```
2. **Sincronización de resultados desde el servidor a tu máquina local:**
   *(El servidor `chome` no cuenta con `rsync`, por lo que se recomienda transferir comprimido vía `ssh` + `tar` o con `scp`)*:
   ```bash
   # Traer figuras, resultados y logs comprimidos al vuelo
   ssh grupo12@chome.inf.udec.cl "tar -czf - -C ~/BCP-K-Optimization outputs/figures outputs/results outputs/logs" | tar -xzf -

   # Alternativa directa con scp (incluyendo opcionalmente una carpeta de ejecución específica)
   scp -r \
     grupo12@chome.inf.udec.cl:~/BCP-K-Optimization/outputs/figures \
     grupo12@chome.inf.udec.cl:~/BCP-K-Optimization/outputs/results \
     grupo12@chome.inf.udec.cl:~/BCP-K-Optimization/outputs/logs \
     ./outputs/
   ```
