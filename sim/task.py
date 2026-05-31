'''
task defines warehouse tasks which are created by task master and stored in warehouse queue.
'''

import numpy as np

class Task:
    # simple task consisting of travel, work and return.

    _task_id = 0                                                                            # global task id

    def __init__(self, station, duration, created_at):
        Task._task_id += 1
        self.id = Task._task_id
        self.station = station                                                              # location of the task
        self.duration = duration                                                            # mins to complete the task
        self.created_at = created_at                                                        # ref. task creation time (for queueing)

class TaskGenerator:
    # Poisson arrival process since inter-arrival is exponential. puts tasks into warehouse.task_queue.

    def __init__(self, env, config, warehouse):
        self.env = env
        self.config = config
        self.warehouse = warehouse
        self.rng = np.random.default_rng(config["seed"] + 1)                                # 1 is some arbitrary offset to be different that warehouse rng generator

    def generate(self):
        # this generator yields inter-arrival times and creates tasks indefinitely.

        while True:
            inter_arrival = self.rng.exponential(1.0 / self.config["task_arrival_rate"])    # stochastic exponential arrival time (Poisson process)
            yield self.env.timeout(inter_arrival)                                           # suspend until arrival time
            station = self.warehouse.stations[self.rng.integers(len(self.warehouse.stations))]  # select one of stations randomly
            duration = self.rng.uniform(1.0, 5.0)                                           # select task duration random at uniform. in reality has to be empirically measured
            task = Task(station, duration, self.env.now)                                    # create the task
            self.warehouse.task_queue.put(task)                                             # feed to queue
            self.warehouse.metrics.log_task_created(task)                                   # record task creation for performance analysis


