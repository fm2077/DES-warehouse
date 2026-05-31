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
        while True:
            #-----------idle------------
            self.metrics.log_robot_status(self.id, "idle", self.env.now)
            task = yield self.warehouse.task_queue.get()                                    # task is the event object that yield returns

            #----------travel-----------
            self.metrics.log_robot_status(self.id, "traveling", self.env.now)
            traversal = self.warehouse.travel_time(self.warehouse.home, task.station)
            yield self.env.timeout(traversal)

            #---------execution---------
            self.metrics.log_robot_status(self.id, "working", self.env.now)
            yield self.env.timeout(task.duration)

            #----------return-----------
            self.metrics.log_robot_status(self.id, "returning", self.env.now)
            returning = self.warehouse.travel_time(task.station, self.warehouse.home)
            yield self.env.timeout(returning)
            self.metrics.log_task_completed(task, self.id, self.env.now)





