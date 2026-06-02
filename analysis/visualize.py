import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


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


def plot_warehouse(warehouse):
    # visualizes the warehouse grid, obstacles, stations, and home.

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for (u, v) in warehouse.graph.edges():                                                  # draw grid edges
        ax.plot([u[0], v[0]], [u[1], v[1]], color="lightgray", linewidth=0.5)

    for obs in warehouse.obstacles:                                                         # draw obstacles
        ax.add_patch(plt.Rectangle((obs[0] - 0.4, obs[1] - 0.4), 0.8, 0.8,
                                    color="black", zorder=3))

    sx = [s[0] for s in warehouse.stations]                                                 # draw stations
    sy = [s[1] for s in warehouse.stations]
    ax.scatter(sx, sy, color="steelblue", s=120, zorder=4, label="stations")

    cx = [cs[0] for cs in warehouse.charging_stations]                                      # charging stations
    cy = [cs[1] for cs in warehouse.charging_stations]
    ax.scatter(cx, cy, color="gold", s=75, zorder=5, label="stations", marker="D")

    ax.scatter(*warehouse.home, color="red", s=150, marker="s", zorder=5, label="home")     # draw home

    for station in warehouse.stations:                                                      # draw a sample path from home to each station
        path = warehouse.find_path(warehouse.home, station)
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, linewidth=1.5, alpha=0.6, linestyle="--")

    ax.set_xlim(-1, warehouse.grid_size[0]); ax.set_ylim(-1, warehouse.grid_size[1])
    ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_title("Warehouse Grid Layout")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("warehouse_layout.png", dpi=150)
    plt.show()


from matplotlib.animation import FuncAnimation
import pandas as pd


def animate_warehouse(warehouse, metrics, config, speedup=10, interval=50):
    """
    Replays simulation as animated robots on the grid.
    speedup: how many sim-minutes per real second
    interval: ms between frames
    """

    df = pd.DataFrame(metrics.robot_positions, columns=["robot_id", "x", "y", "time"])

    sim_duration = config["sim_duration"]
    dt = speedup * interval / 400                                                          # sim-time step per frame
    frames = np.arange(0, sim_duration, dt)
    num_robots = config["num_robots"]

    # ───── precompute positions for all frames ──────
    position_cache = np.zeros((len(frames), num_robots, 2))
    for rid in range(num_robots):
        robot_df = df[df["robot_id"] == rid].sort_values("time")
        times = robot_df["time"].values
        xs = robot_df["x"].values
        ys = robot_df["y"].values
        for i, t in enumerate(frames):
            idx = np.searchsorted(times, t, side="right") - 1                               # binary search for speed up
            if idx >= 0:
                position_cache[i, rid] = [xs[idx], ys[idx]]                                 # cache for speed up
            else:
                position_cache[i, rid] = warehouse.home

    # ──────figure ────────
    fig, ax = plt.subplots(figsize=(8, 8))

    # ──────static elements──────
    for (u, v) in warehouse.graph.edges():
        ax.plot([u[0], v[0]], [u[1], v[1]], color="lightgray", linewidth=0.5)

    for obs in warehouse.obstacles:                                                         # obstacles
        ax.add_patch(plt.Rectangle((obs[0] - 0.4, obs[1] - 0.4), 0.8, 0.8,
                                    color="black", zorder=3))

    sx = [s[0] for s in warehouse.stations]                                                 # work stations
    sy = [s[1] for s in warehouse.stations]
    ax.scatter(sx, sy, color="steelblue", s=100, zorder=4, label="stations", marker="^")
    ax.scatter(*warehouse.home, color="red", s=120, marker="s", zorder=4, label="home")

    cx = [cs[0] for cs in warehouse.charging_stations]                                      # charging stations
    cy = [cs[1] for cs in warehouse.charging_stations]
    ax.scatter(cx, cy, color="gold", s=75, zorder=5, label="stations", marker="D")

    ax.set_xlim(-1, warehouse.grid_size[0]); ax.set_ylim(-1, warehouse.grid_size[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    # ──────dynamic elements──────
    robot_dots = ax.scatter([], [], color="orange", s=80, zorder=5, edgecolors="black", linewidths=0.5)
    time_text = ax.set_title("")

    # ─────animation loop───────
    def update(frame_idx):
        robot_dots.set_offsets(position_cache[frame_idx])
        ax.set_title(f"t = {frames[frame_idx]:.1f} min")
        return robot_dots,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True, repeat=False)

    anim.save("warehouse_animation.gif", writer="pillow", fps=400 // interval)
    print("Saved warehouse_animation.gif")
    plt.show()
