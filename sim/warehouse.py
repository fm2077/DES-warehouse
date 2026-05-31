'''
warehouse hosts the environment for grid layout, station locations and shared task queue.
'''

import simpy
import numpy as np

class Warehouse:
    def __init__(self, env, config, metrics):
        self.env = env
        self.config = config
        self.metrics = metrics
        self.grid_size = (20, 20)
        self.home = (0,0)                                                                   # robot start/return position

        rng = np.random.default_rng(config["seed"])                                         # initiate an rng generator
        self.stations = [
            tuple(rng.integers(1, self.grid_size[0], size=2))                               # random station location x,y
            for _ in range(config["num_stations"])
        ]
        self.task_queue = simpy.Store(env)                                                  # default unlimited capacity FIFO queue storing the tasks. Robot and task master inherit task_queue from Warehouse

    def travel_time(self, pos_from, pos_to):
        # uses Manhattan distance (L1 norm) to calculate traversal time over a cartesian grid
        dist = abs(pos_from[0] - pos_to[0]) + abs(pos_from[1] - pos_to[1])
        return dist / self.config["robot_speed"]
