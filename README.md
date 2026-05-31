# DES-warehouse

This is a Discrete-Event Simulation (DES) planner for a multi-agent environment where a fleet of robots perform warehouse tasks to optimize aggregate throughput.\
The queue management is simple FIFO. Current version has no collision modeling, charge time, obstacles or path planning.

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
│   └── visualize.py
│
├── tests/
│   └── test_sim.py
│
├── main.py
├── config.json
├── results.png
├── requirements.txt
├── .gitignore
└── README.md
````

## Stack
- `simpy` to simulate environment
- `numpy`, `pandas` for data analysis and statistical analysis
- `scipy` for optimization
- `pytest` for testing
- `matplotlib` for visualization

## Usage
the simulation input can be adjusted in `config.json`.

to run a simulation:
```bash
python main.py
```

to run tests:
```bash
pytest tests/test_sim.py -v
````