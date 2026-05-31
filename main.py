import simpy
import json
from sim.warehouse import Warehouse
from sim.task import TaskGenerator
from sim.robot import Robot
from analysis.metrics import MetricsCollector
from analysis.visualize import plot_results

with open("config.json") as f:
    CONFIG = json.load(f)                                                                   # load simulation input data

def run_sim(config):
    # receives the DES config, runs the simulation and returns the performance metrics.

    env = simpy.Environment()                                                               # instantiate an environment
    metrics_sim = MetricsCollector(env)                                                     # instantiate a logger
    warehouse = Warehouse(env, config, metrics_sim)                                         # instantiate a warehouse
    task_master = TaskGenerator(env, config, warehouse)                                     # instantiate a task generator
    env.process(task_master.generate())                                                     # register the task master with the queue and wrap it as a process

    for i in range(config["num_robots"]):                                                   # run the entire fleet of robots for the duration
        robot = Robot(env, i, config, warehouse, metrics_sim)                               # pull the robot and instantiate it
        env.process(robot.work())                                                           # register each robot generator with the queue and wrap it as a process
    env.run(until = config["sim_duration"])

    return metrics_sim

if __name__ == "__main__":
    metrics_sim = run_sim(CONFIG)
    metrics_sim.summary(CONFIG["num_robots"])
    plot_results(metrics_sim, CONFIG["num_robots"])
