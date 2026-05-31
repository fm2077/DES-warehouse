import numpy as np


class MetricsCollector:
    def __init__(self, env):
        self.env = env
        self.tasks_created = []                                                             # some event logs
        self.tasks_completed = []
        self.robot_status_log = []


    def log_task_created(self, task):
        self.tasks_created.append((task.id, task.created_at, task.station))


    def log_task_completed(self, task, robot_id, completed_at):
        self.tasks_completed.append((task.id, robot_id, task.created_at, completed_at))


    def log_robot_status(self, robot_id, status, timestamp):
        self.robot_status_log.append((robot_id, status, timestamp))


    def throughput(self):
        # run this after each completion, over time converges to expected throughput
        if not self.tasks_completed:
            return 0.0
        total_time = self.env.now
        return len(self.tasks_completed) / total_time
    

    def latencies(self):
        return np.array([completed_at - created_at
                         for _, _, created_at, completed_at in self.tasks_completed])       # task creation to completion for each task
    

    def utilization(self, num_robots):
        # fraction of non-idle robot time
        util = {}
        if not self.robot_status_log:
            return util
        for robot_id in range(num_robots):
            timeline = [
                (ts, status) for rid, status, ts in self.robot_status_log if rid == robot_id
            ]                                                                               # get time stamps and status for each robot in ascending order
            timeline.sort(key = lambda x: x[0])                                             # safeguard for chronological order sorting. already should be sorted by timestamp

            active_time = 0.0
            for i in range(len(timeline) - 1):
                ts, status = timeline[i]
                next_ts = timeline[i+1][0]
                if status != "idle":
                    active_time += next_ts - ts

            if timeline:                                                                    # last status to end of simulation is edge case. can be safely dropped for long enough simulation
                last_ts, last_status = timeline[-1]
                if last_status != "idle":
                    active_time += self.env.now - last_ts

            util[robot_id] = active_time / self.env.now if self.env.now > 0 else 0.0

        return util
    

    def summary(self, num_robots):
        lat = self.latencies()
        util = self.utilization(num_robots)        
        print(f"  DES SUMMARY  (T = {self.env.now:.1f} min)")
        print(f"  Tasks created:     {len(self.tasks_created)}")
        print(f"  Tasks completed:   {len(self.tasks_completed)}")
        print(f"  Throughput:        {self.throughput():.2f} tasks/min")
        print(f"  Avg latency:       {np.mean(lat):.2f} min")
        print(f"  Max latency:       {np.max(lat):.2f} min")
        print(f"  Avg utilization:   {np.mean(list(util.values())):.1%}")

    