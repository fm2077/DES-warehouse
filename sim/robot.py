'''
robot agent gets task from task master, travels to station, executes, returns and repeats.
simple model without collision modeling, charge time, and no path planning.
'''

import networkx as nx


class Robot:
    def __init__(self, env, robot_id, config, warehouse, metrics):
        self.env = env
        self.id = robot_id
        self.config = config
        self.warehouse = warehouse
        self.metrics = metrics

    def work(self):
        self.pos = self.warehouse.home                                                      # default position is home, espeically for fifo case
        while True:
            #-----------idle------------
            self.metrics.log_robot_status(self.id, "idle", self.env.now)
            self.metrics.log_robot_position(self.id, self.pos, self.env.now)
            task = yield self.env.process(
                self.warehouse.assign_task(self.pos, policy=self.config["policy"])          # generic task assignment based on input policy
            )                                    

            #----------travel-----------
            self.metrics.log_robot_status(self.id, "traveling", self.env.now)
            prev_req = None                                                                 # to track occupied cells
            path = self.warehouse.find_path(self.pos, task.station)                         # get travel path from motion planner            

            while self.pos != task.station:
                for step in path[1:]:
                    req = self.warehouse.cell_locks[step].request()                         # ask to use the resource (1 occupancy)
                    timeout = self.env.timeout(self.config["deadlock_timeout"])

                    result = yield req | timeout                                            # suspend until next cell becomes available (request fulfilled) or if a deadlock times out
                    if req in result:                                                       # proceed normally
                        if prev_req is not None:
                            self.warehouse.cell_locks[self.pos].release(prev_req)           # free the resource (cell left behind)
                        yield self.env.timeout(1.0 / self.config["robot_speed"])            # every single grid step suspend to check for collision and log position
                        self.pos = step
                        self.metrics.log_robot_position(self.id, self.pos, self.env.now)
                        prev_req = req                                                      # carry forward to release next step
                    else:                                                                   # if trapped deadlock, cancel request and reroute
                        req.cancel()
                        new_path = self._reroute(task.station, step)                        # reroute                                                  
                        if new_path is not None:
                            path = new_path
                        else:
                            yield self.env.timeout(self.config["deadlock_timeout"])         # if no alternative path available, wait another timeout and retry again
                            path = self.warehouse.find_path(self.pos, task.station)
                        break                                                               # once reroute found restart travel with a new path

            if prev_req is not None:
                self.warehouse.cell_locks[self.pos].release(prev_req)                       # free final station cell when arrived

            #---------execution---------
            self.metrics.log_robot_status(self.id, "working", self.env.now)
            yield self.env.timeout(task.duration)

            #----------return-----------
            '''
            self.metrics.log_robot_status(self.id, "returning", self.env.now)
            return_path = self.warehouse.find_path(self.pos, self.warehouse.home)           # get return travel path from motion planner
            for step in return_path[1:]:
                yield self.env.timeout(1.0 / self.config["robot_speed"])                    # every single grid step suspend to check for collision and log position
                self.pos = step
                self.metrics.log_robot_position(self.id, self.pos, self.env.now)
            self.metrics.log_task_completed(task, self.id, self.env.now)
            '''

            #----------no return, stay at station-----------
            self.metrics.log_task_completed(task, self.id, self.env.now)                    # switched to stay at station so greedy policy makes sense


    def _reroute(self, target, blocked_cell):
        # finds an alternative path avoiding an occupied cell. Returns none if no alt. exists.

        temp_graph = self.warehouse.graph.copy()
        if blocked_cell in temp_graph:
            temp_graph.remove_node(blocked_cell)
        try:
            return nx.astar_path(temp_graph, self.pos, target, heuristic=self.warehouse.heuristic)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None





