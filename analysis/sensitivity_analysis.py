'''
Performs Sobol Sensitivity Analysis to identify which input parameters impact the throughput output the most. Runs independently from the simulation engine.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from SALib.sample import sobol as sobol_sampler                                             # Sobol Saltelli estimator
from SALib.analyze import sobol
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed                            # high level multi-process. As completed returns in order of completion due to asymmetric loading
import sys
sys.path.insert(0, ".")
from sim.task import Task
from main import run_sim


with open("config.json") as f:
    REF_CONFIG = json.load(f)


#─────────PARAM SPACE──────────
# Some parameters left out as they are fixed such as grid size, randomization seed, sim_duration
# some are categorical parameters such as policy
PROBLEM = {                                                                                 # Saltelli default input name
    "num_vars": 13,                                                                         # do not change key names, Saltelli expects defaults
    "names": [
        "num_robots",
        "num_stations",
        "task_arrival_rate",
        "robot_speed",
        "grid_dim_1",
        "grid_dim_2",
        "deadlock_timeout",
        "battery_capacity",
        "battery_drain_travel",
        "battery_drain_work",
        "battery_threshold",
        "charge_rate",
        "num_chargers"
    ],
    "bounds": [
        [3,     25],                                                                        # num_robots
        [5,     20],                                                                        # num_stations
        [0.5,  5.0],                                                                        # task_arrival_rate
        [0.5,  2.0],                                                                        # robot_speed
        [10,    30],                                                                        # grid_dim_1
        [10,    30],                                                                        # grid_dim_2
        [1.0, 10.0],                                                                        # deadlock_timeout
        [50.0, 150.0],                                                                      # battery_capacity
        [0.5,  4.0],                                                                        # battery_drain_travel
        [1.0,  8.0],                                                                        # battery_drain_work
        [10.0,40.0],                                                                        # battery_threshold
        [2.0, 10.0],                                                                        # charge_rate
        [1,      4]                                                                         # num_chargers
    ]
}
INT_PARAMS = {"num_robots", "num_stations", "num_chargers", "grid_dim_1", "grid_dim_2"}     # have to be rounded before sim

#─────────SENSITIVITY ANALYSIS──────────

def build_config(problem: dict, sample: np.ndarray, ref_config: dict) -> dict:
# extract a config from the Sobol sample

    config = ref_config.copy()                                                              # explicitly provide and use ref config for parallel reliability
    for i, name in enumerate(problem["names"]):
        val = sample[i]
        if name in INT_PARAMS:
            val = int(round(val))                                                           # override if integer parameter
        config[name] = val
    return config


def _run_single(args: tuple) -> tuple[int, float]:
# Single process simulation for parallel mode, returning the index and throughput

    i, sample, problem, ref_config = args
    Task._task_id = 0
    config = build_config(problem, sample, ref_config)                                      # explicitly passing reference config for parallel reliability
    try:
        metrics, _ = run_sim(config)
        return i, metrics.throughput()                                                      # pass the index with throughput in parallel mode
    except Exception as e:
        print(f"run {i} failed: {e}, setting throughput to 0.0")
        return i, 0.0


def run_sobol(problem: dict, N: int = 64, n_workers: int = None) -> dict:
# complexity is N(num_var+2) for Saltelli estimator. N<100 acceptable for case study, N>500 for production grade.
# N should be power of 2. Cost scales linearly with parameter space. Returns a dictionary of Sobol indices.
# runs in parallel mode.

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)                                              # leave out one core for OS

    print(f"Generating Saltelli samples: N = {N}, Total Sims = {N*(problem['num_vars']+2)}, Workers={n_workers}")
    par_val = sobol_sampler.sample(problem, N=N, calc_second_order=False)                   # sample from parameter space and skip second order indices. Uses LHS under the hood.
    Y = np.zeros(len(par_val))                                                              # initiate output vector

    args = [(i, sample, problem, REF_CONFIG) for i, sample in enumerate(par_val)]           # form input to each process, the simulation index should be passed for traceback
    with ProcessPoolExecutor(max_workers = n_workers) as executor:                          # multi-process wrapper
        futures = {executor.submit(_run_single, arg): arg[0] for arg in args}               # futures objects are promises of results for submitted tasks. A dictionary of future object key and index value
        for future in tqdm(as_completed(futures),                                           # yield futures in order of completion not submission, no idling and accurate tqdm progress
                            total=len(par_val),
                            desc="Running simulations",
                            unit="sim"):                                                    # wrap simulations with tqdm for time tracking
            i, throughput = future.result()                                                 # pick up results and unblock the process
            Y[i] = throughput
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)         # calculate first order and total order Sobol indices
    return Si


def plot_sobol(problem: dict, Si: dict) -> None:
    names = problem["names"]
    S1 = Si["S1"]; S1_conf = Si["S1_conf"]                                                  # get indices and their resampling bootstrapped 95% confidence intervals
    ST = Si["ST"]; ST_conf = Si["ST_conf"]    

    x = np.arange(len(names))                                                               # bar chart bins
    width = 0.35                                                                            # bar chart width

    fig, ax = plt.subplots(figsize=(12,5))
    barS1 = ax.bar(x - width/2, S1, width, label="S1 (first order)",
                   color="steelblue", yerr=S1_conf, capsize=4)
    barST = ax.bar(x + width/2, ST, width, label="ST (total order)",
                   color="coral", yerr=ST_conf, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Sobol Indices")
    ax.set_title("Sobol Sensitivity Analysis — Throughput")
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="significance threshold (0.05)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("sobol_indices.png", dpi=150)
    plt.show()


def print_summary(problem: dict, Si: dict) -> None:
    names = problem["names"]
    S1 = Si["S1"]; ST = Si["ST"]
    S1_conf = Si["S1_conf"]; ST_conf = Si["ST_conf"]

    print(f"  SOBOL SENSITIVITY RESULTS")
    print(f"  {'Parameter':<22} {'S1':>8} {'ST':>8} {'Interaction':>12}")

    ranked_names = sorted(zip(names, S1, ST, S1_conf, ST_conf), key=lambda x: x[2], reverse=True)  # rank by total order Sobol index
    for name, s1, st, s1c, stc in ranked_names:
        interaction = st - s1
        print(f" {name:<22} {s1:>8.3f} {st:>8.3f} {interaction:>12.3f}")

    insignificant = [name for name, s1, st, s1c, stc in ranked_names if st < 0.05]          # 5% threshold based on sensitivity analysis convention
    if insignificant:
        print(f"Parameters with ST < 0.05, consider leaving out of simulation:")
        for name in insignificant:
            print(f" - {name}")
    else:
        print("All considered parameters have meaningful total order influence.")
    _save_excel(ranked_names)                                                               # save sensitivity results to to XLSX file


def _save_excel(ranked: list) -> None:
    df = pd.DataFrame(
        [(name, s1, st, st - s1, s1c, stc) for name, s1, st, s1c, stc in ranked],
        columns = ["Parameter", "S1", "ST", "Interaction", "S1_conf", "ST_conf"]
    )
    df["Significant"] = (df["ST"] >= 0.05)                                                  # add a significance column
    df = df.round(4)
    df.to_excel("sobol_results.xlsx", index=False)
    print("\nSaved: sobol_results.xlsx")


if __name__ == "__main__":                                                                  # mandatory since this is parallel
    Si = run_sobol(PROBLEM, N=512)                                                          # can explicitly provide n_workers
    print_summary(PROBLEM, Si)
    plot_sobol(PROBLEM, Si)
