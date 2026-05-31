import matplotlib.pyplot as plt
import numpy as np


def plot_results(metrics, num_robots):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lats = metrics.latencies()
    axes[0].hist(lats, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(np.mean(lats), color="red", linestyle="--", label=f"mean = {np.mean(lats):.1f}")
    axes[0].set_xlabel("Latency (min)"); axes[0].set_ylabel("Count"); axes[0].set_title("Task Latency Distribution")
    axes[0].legend()

    util = metrics.utilization(num_robots)
    robot_ids = list(util.keys()); util_vals = list(util.values())
    axes[1].bar(robot_ids, util_vals, color="teal", edgecolor="white")
    axes[1].axhline(np.mean(util_vals), color="red", linestyle="--", label=f"mean = {np.mean(util_vals):.1%}")
    axes[1].set_xlabel("Robot ID"); axes[1].set_ylabel("Utilization"); axes[1].set_title("Robot Utilization"); axes[1].set_ylim(0, 1)
    axes[1].legend()

    completed_times = [completed_at for _, _, _, completed_at in metrics.tasks_completed]
    completed_times.sort()
    cumulative = np.arange(1, len(completed_times) + 1)
    axes[2].plot(completed_times, cumulative, color="darkgreen")
    axes[2].set_xlabel("Sim Time (min)"); axes[2].set_ylabel("Cumulative Tasks Completed"); axes[2].set_title("Throughput Over Time")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
