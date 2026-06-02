'''
warehouse hosts the environment for grid layout, station locations, motion planning and shared task queue.
'''

import simpy
import numpy as np
import networkx as nx

class Warehouse:
    def __init__(self, env, config, metrics):
        self.env = env
        self.config = config
        self.metrics = metrics
        self.grid_size = (20, 20)
        self.home = (0,0)                                                                   # robot start/return position

        self.graph = nx.grid_2d_graph(self.grid_size[0], self.grid_size[1])                 # let nx to manage grid for motion planning        

        self.obstacles = [(5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7), 
                          (11, 6), (11, 7), (11, 8), (12, 8), (13, 8)]                      # obstacles
        for obs in self.obstacles:
            if obs in self.graph:
                self.graph.remove_node(obs)

        self.cell_locks = {n: simpy.Resource(env, capacity=1) for n in self.graph.nodes}    # lock occupied cells to avoid collision

        rng = np.random.default_rng(config["seed"])                                         # initiate an rng generator
        valid_nodes = list(self.graph.nodes)                                                # exclude obstacles
        valid_nodes.remove(self.home)                                                       # exclude home from stations pool
        station_indices = rng.choice(len(valid_nodes), size=config["num_stations"], replace=False) # random stations
        self.stations = [valid_nodes[i] for i in station_indices]                           # set location of stations

        #self.task_queue = simpy.Store(env)                                                 # default unlimited capacity FIFO queue storing the tasks. Robot and task master inherit task_queue from Warehouse
        self.task_queue = simpy.FilterStore(env)                                            # use greedy policy instead of simple FIFO queue
        self.waiting_robots = []                                                            # keep track of idle robots so only one of them is notified when task is available

    @staticmethod
    def heuristic(a, b):                                                                    # L1 is a perfect heuristic for this grid as it is both admissible and consistent
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    

    def find_path(self, pos_from, pos_to):
        return nx.astar_path(self.graph, pos_from, pos_to, heuristic=self.heuristic)        # use simple A-star with L1 heuristics for motion planning


    def travel_time(self, pos_from, pos_to):
        path = self.find_path(pos_from, pos_to)                                             # use A-star for optimal path
        dist = len(path) - 1                                                                # num edges
        return dist / self.config["robot_speed"]


    def notify_task_available(self):
        # wakes up a single nearest robot waiting for a task and assigns the task to it.

        if not self.waiting_robots:                                                         # if no robot in the pool, task waits in queue
            return
        
        task = self.task_queue.items[-1]                                                    # get newest task
        if self.config["policy"] == "fifo":
            robot_event, robot_pos = self.waiting_robots.pop(0)                             # if FIFO policy assign to the longest waiting robot
            robot_event.succeed(value=task)                                                 # pass the task directly to waking robot
            self.task_queue.items.remove(task)
        elif self.config["policy"] == "greedy":
            best_idx = min(range(len(self.waiting_robots)),
                           key = lambda i : self.heuristic(self.waiting_robots[i][1], task.station))
            robot_event, robot_pos = self.waiting_robots.pop(best_idx)
            robot_event.succeed(value=task)
            self.task_queue.items.remove(task)


    def assign_task(self, robot_pos, policy='fifo'):
        # yields until dispatched. Returns task.

        while True:
            if self.task_queue.items and not self.waiting_robots:
            # if tasks available but no robots available, there is no competition, 
            # so assignment is robot-centered not task-centered
                if policy == 'fifo':                                                        # first in first out
                    task = self.task_queue.items.pop(0)
                elif policy == 'greedy':                                                    # greedy selects the closest task
                    task = min(self.task_queue.items,
                                key = lambda t : self.heuristic(robot_pos, t.station))
                    self.task_queue.items.remove(task)
                return task
            else:
                wait_event = self.env.event()                                               # create an event object for this robot
                self.waiting_robots.append((wait_event, robot_pos))                         # register this event with other idle robots and its position for greedy selection
                task = yield wait_event                                                     # suspend this robot until a notification
                if task is not None:
                    return task