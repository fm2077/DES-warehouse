# DES-warehouse

This is a Discrete-Event Simulation (DES) planner for a multi-agent environment where a fleet of robots perform warehouse tasks to optimize aggregate throughput.

## Project Layout

DES-warehouse/
├── sim/                                                                                    # core scripts
│   ├── __init__.py
│   ├── warehouse.py                                                                        # environment
│   ├── robot.py                                                                            # agent
│   └── task.py                                                                             # planner
│
├── analysis/                                                                               # performance scripts
│   ├── __init__.py
│   ├── metrics.py
│   └── visualize.py
│
├── tests/
│   └── test_sim.py
│
├── main.py                                                                                 # main entry point, orchestrator
├── requirements.txt
├── .gitignore
└── README.md

## Stack
- simpy, to simulate environment
- numpy, pandas, for data analysis and statistical analysis
- scipy, for optimization
- pytest, for testing
- matplotlib, for visualization