# DES-warehouse

This is a Discrete-Event Simulation (DES) planner for a multi-agent environment where a fleet of robots perform warehouse tasks to optimize aggregate throughput.\
The input parameters are sensitivity-analyzed using Sobol indices in parallel and immaterial parameters are marked to shrink the parameter space.\
The queue assignment policy is either FIFO or greedy. Current version includes obstacles, path planning using A-star, cell-lock collision management with deadlock resolution via reroute and reactive battery charging.\
Greedy assignment is task-centered selecting nearest robot to the task. A Monte Carlo statistical analysis analyzes whether greedy policy has significant advantage over simple FIFO.

## Project Layout

````
DES-warehouse/
├── sim/
│   ├── __init__.py
│   ├── warehouse.py
│   ├── robot.py
│   └── task.py
│
├── analysis/
│   ├── __init__.py
│   ├── metrics.py
│   ├── monte_carlo.py
│   ├── sensitivity_analysis.py
│   └── visualize.py
│
├── tests/
│   └── test_sim.py
│
├── main.py
├── config.json
├── monte_carlo_results.png
├── results.png
├── sobol_indices.png
├── sobol_results.xlsx
├── warehouse_animation.gif
├── warehouse_layout.png
├── requirements.txt
├── .gitignore
└── README.md
````

## Stack
- `simpy` to simulate environment
- `numpy`, `pandas` for data analysis and stochastic modeling
- `scipy` for statistical analysis
- `networkx` for grid construction and motion planning
- `SALib` for sensitivity analysis
- `tqdm` for simulation time tracking
- `pytest` for testing
- `matplotlib`, `openpyxl` for visualization & tabulation

## Usage
the simulation input can be adjusted in `config.json`.

to run a simulation:
```bash
python main.py
```

to run tests:
```bash
pytest tests/test_sim.py -v
```

to run sensitivity analysis:
```bash
python -m analysis.sensitivity_analysis
```

to run Monte Carlo statistical analysis:
```bash
python -m analysis.monte_carlo
````