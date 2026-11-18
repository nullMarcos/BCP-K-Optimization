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
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Pruebas unitarias de restricciones y factibilidad
│   └── test_callbacks.py     # Pruebas unitarias para el callback de componentes conexas
├── outputs/
│   ├── logs/                 # Archivos de registro (.log) de Gurobi y CPLEX
│   ├── results/              # Archivos CSV consolidados (tiempos, gaps, aristas cortadas)
│   └── figures/              # Gráficos y curvas trade-off generados
├── .gitignore
├── README.md
└── requirements.txt
