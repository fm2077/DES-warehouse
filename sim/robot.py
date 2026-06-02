'''
robot agent gets task from task master, travels to station, executes, returns and repeats.
includes motion planning, collision avoidance with cell locks, deadlock resolution, and reactive battery charging.
'''

import networkx as nx


class Robot:
    def __init__(self, env, robot_id, config, warehouse, metrics):
        self.env = env
        self.id = robot_id
        self.config = config
        self.warehouse = warehouse
        self.metrics = metrics
        self.battery = config["battery_capacity"]


    def _travel(self, destination):
        # refactored from work so both charge and work can call it

        prev_req = None                                                                     # to track occupied cells
        path = self.warehouse.find_path(self.pos, destination)                              # get travel path from motion planner            

        while self.pos != destination:
            for step in path[1:]:
                req = self.warehouse.cell_locks[step].request()                             # ask to use the resource (1 occupancy)
                timeout = self.env.timeout(self.config["deadlock_timeout"])
                result = yield req | timeout                                                # suspend until next cell becomes available (request fulfilled) or if a deadlock times out
                if req in result:                                                           # proceed normally
                    if prev_req is not None:
                        self.warehouse.cell_locks[self.pos].release(prev_req)               # free the resource (cell left behind)
                    yield self.env.timeout(1.0 / self.config["robot_speed"])                # every single grid step suspend to check for collision and log position
                    self.pos = step
                    self.battery -= self.config["battery_drain_travel"]                     # how much battery was drained during travel (fixed number simple model)
                    self.metrics.log_robot_position(self.id, self.pos, self.env.now)
                    prev_req = req                                                          # carry forward to release next step
                else:                                                                       # if trapped deadlock, cancel request and reroute
                    req.cancel()
                    new_path = self._reroute(destination, step)                             # reroute                                                  
                    if new_path is not None:
                        path = new_path
                    else:
                        yield self.env.timeout(self.config["deadlock_timeout"])             # if no alternative path available, wait another timeout and retry again
                        path = self.warehouse.find_path(self.pos, destination)
                    break                                                                   # once reroute found restart travel with a new path

        if prev_req is not None:
            self.warehouse.cell_locks[self.pos].release(prev_req)                           # free final station cell when arrived    
    
    
    def charge(self):
        # travels to nearest charger, wait for empty charger, recharges completely and releases.

        nearest = min(self.warehouse.charging_stations,
                      key=lambda cs: self.warehouse.heuristic(self.pos, cs))                # find the nearest charger
        self.metric.log_robot_status(self.id, "traveling_to_charger", self.env.now)
        yield self.env.process(self._travel(nearest))                                       # suspend until arrival at charging station

        self.metrics.log_robot_status(self.id, "charging", self.env.now)
        req = self.warehouse.charger_resources[nearest].request()
        yield req                                                                           # wait until a slot is available for charging

        charge_time = (self.config["battery_capacity"] - self.battery) / self.config["charge_rate"] # wait until fully charged
        yield self.env.timeout(charge_time)
        self.battery = self.config["battery_capacity"]
        self.warehouse.charger_resources[nearest].release(req)                              # release the charging station
        self.metrics.log_robot_status(self.id, "charged", self.env.now)
        self.metrics.log_charging(self.id, self.env.now, self.battery)                      # log charge event


    def work(self):
        self.pos = self.warehouse.home                                                      # default position is home, especially for fifo case
        while True:
            #-----------idle------------
            self.metrics.log_robot_status(self.id, "idle", self.env.now)
            self.metrics.log_robot_position(self.id, self.pos, self.env.now)
            task = yield self.env.process(
                self.warehouse.assign_task(self.pos, policy=self.config["policy"])          # generic task assignment based on input policy
            )                                    

            #----------travel-----------
            self.metrics.log_robot_status(self.id, "traveling", self.env.now)
            yield self.env.process(self._travel(task.station))                              # travel to station, details of mechanics refactored in _travel

            #---------execution---------
            self.metrics.log_robot_status(self.id, "working", self.env.now)
            work_drain = self.config["battery_drain_work"] * task.duration
            yield self.env.timeout(task.duration)
            self.battery -= work_drain                                                      # decharge battery

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

        if blocked_cell == target:                                                          # for good measure, the except should capture this
            return None
        
        temp_graph = self.warehouse.graph.copy()
        if blocked_cell in temp_graph:
            temp_graph.remove_node(blocked_cell)
        try:
            return nx.astar_path(temp_graph, self.pos, target, heuristic=self.warehouse.heuristic)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None





