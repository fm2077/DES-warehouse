'''
robot agent gets task from task master, travels to station, executes, returns and repeats.
simple model without collision modeling, charge time, and no path planning.
'''

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
            path = self.warehouse.find_path(self.pos, task.station)                         # get travel path from motion planner
            for step in path[1:]:
                yield self.env.timeout(1.0 / self.config["robot_speed"])                    # every single grid step suspend to check for collision and log position
                self.pos = step
                self.metrics.log_robot_position(self.id, self.pos, self.env.now)

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





