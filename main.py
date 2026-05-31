import simpy
from sim.warehouse import Warehouse
from sim.task import TaskGenerator
from sim.robot import Robot
from analysis.metrics import MetricsCollector
from analysis.visualize import plot_results

CONFIG = {
    "num_robots": 10,
    "num_stations": 5,                                                                      # pick up and drop off locations
    "sim_duration": 120,                                                                    # in min
    "task_arrival_rate": 2.0,                                                               # is task/min, lambda, used in Poisson arrival time calc.
    "robot_speed": 1.0,                                                                     # in grid_units/min
    "seed": 42                                                                              # for reproducibility and debugging
}

def run_sim(config):
    # receives the DES config, runs the simulation and returns the performance metrics.

    env = simpy.Environment()                                                               # instantiate an environment
    metrics = MetricsCollector(env)                                                         # instantiate a logger
    warehouse = Warehouse(env, config, metrics)                                             # instantiate a warehouse
    task_master = TaskGenerator(env, config, warehouse)                                     # instantiate a task generator
    env.process(task_master.generate())                                                     # register the task master with the queue and wrap it as a process

    for i in range(config["num_robots"]):                                                   # run the entire fleet of robots for the duration
        robot = Robot(env, i, config, warehouse, metrics)                                   # pull the robot and instantiate it
        env.process(robot.work())                                                           # register each robot generator with the queue and wrap it as a process
    env.run(until = config["sim_duration"])

    return metrics_sim

if __name__ == "__main__":
    metrics_sim = run_sim(CONFIG)
    metrics_sim.summary()
    plot_results(metrics_sim)
