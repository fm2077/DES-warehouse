'''
Monte Carlo sweep for policy optimization.
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json
from sim.task import Task
from main import run_sim


with open("config.json") as f:
    REF_CONFIG = json.load(f)


def sweep(fleet_sizes: list[int], policies: list[str], num_seeds: int = 30) -> dict:
    # sweep over a combination of robot fleet size, policies an d randomization seeds.
    # returns a dictionary of throughputs and latencies for policy and feet size pairs.

    results = {}
    for p in policies:
        results[p] = {}
        for n in fleet_sizes:
            tp = []                                                                         # throughputs
            avg_lat = []                                                                    # mean latencies
            for s in range(num_seeds):
                config = {
                          **REF_CONFIG,
                          "num_robots": n,
                          "policy": p,
                          "seed": s
                          }
                Task._task_id = 0                                                           # reset global task counter
                metrics, warehouse = run_sim(config)
                tp.append(metrics.throughput())
                if len(metrics.latencies()) > 0:
                    avg_lat.append(np.mean(metrics.latencies()))
                else:
                    avg_lat.append(0.0)

            results[p][n] = {
                "throughput": np.array(tp),
                "latency": np.array(avg_lat),
            }
            print(f"  {p:6s} | robots={n:3d} | "
                  f"throughput={np.mean(tp):.2f} ± {np.std(tp):.2f} | "                     # print statistics over a range of rollouts
                  f"latency={np.mean(avg_lat):.1f} ± {np.std(avg_lat):.1f}")
    return results


def plot_sweep(results: dict, fleet_sizes: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"fifo": "steelblue", "greedy": "coral"}

    for p, c in colors.items():
        means_tp = [np.mean(results[p][n]["throughput"]) for n in fleet_sizes]
        stds_tp = [np.std(results[p][n]["throughput"]) for n in fleet_sizes]
        means_lat = [np.mean(results[p][n]["latency"]) for n in fleet_sizes]
        stds_lat = [np.std(results[p][n]["latency"]) for n in fleet_sizes]

        # throughput vs fleet size
        axes[0].plot(fleet_sizes, means_tp, marker='o', color=c, label=p)
        axes[0].fill_between(fleet_sizes,
                             np.array(means_tp) - np.array(stds_tp),
                             np.array(means_tp) + np.array(stds_tp),
                             alpha=0.2, color=c)
        
        # latency vs fleet size
        axes[1].plot(fleet_sizes, means_lat, marker='o', color=c, label=p)
        axes[1].fill_between(fleet_sizes,
                             np.array(means_lat) - np.array(stds_lat),
                             np.array(means_lat) + np.array(stds_lat),
                             alpha=0.2, color=c)
    
    axes[0].set_xlabel("Fleet Size"); axes[0].set_ylabel("Throughput (tasks/min)")
    axes[0].set_title("Throughput vs Fleet Size")
    axes[0].legend()

    axes[1].set_xlabel("Fleet Size"); axes[1].set_ylabel("Mean Latency (min)")
    axes[1].set_title("Latency vs Fleet Size")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("monte_carlo_results.png", dpi=150)
    plt.show()


def compare_policies(results: dict, fleet_size: int) -> None:
    # T-test comparing FIFO vs greedy. Main performance metric is throughput.

    fifo = results["fifo"][fleet_size]["throughput"]                                        # get both policy throughput populations
    greedy = results["greedy"][fleet_size]["throughput"]
    t_stat, p_value = stats.ttest_ind(fifo, greedy)
    print(f"\n  Policy comparison at fleet_size={fleet_size}:")
    print(f"  FIFO   throughput: {np.mean(fifo):.3f} ± {np.std(fifo):.3f}")
    print(f"  Greedy throughput: {np.mean(greedy):.3f} ± {np.std(greedy):.3f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    if p_value < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("Not statistically significant difference (p >= 0.05)")


if __name__ == "__main__":
    fleet_sizes = [3, 5, 8, 10, 15, 20]
    policies = ["fifo", "greedy"]
    print("Running Monte Carlo Sweep ...")
    results = sweep(fleet_sizes, policies, num_seeds=30)
    plot_sweep(results, fleet_sizes)
    compare_policies(results, fleet_size=10)

