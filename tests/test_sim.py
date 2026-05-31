'''
unit tests for DES-warehouse simulation. 
Run: pytest tests/test_sim.py -v
'''

import simpy
import json
import sys
sys.path.insert(0, ".")                                                                     # also look at the root folder "." when running pytest with highest priority
print(sys.path)
from sim.warehouse import Warehouse
from sim.task import Task, TaskGenerator
from sim.robot import Robot
from analysis.metrics import MetricsCollector


with open("config.json") as f:
    CONFIG = json.load(f)

#---------unit tests--------------

def make_env():
    env = simpy.Environment()
    metrics = MetricsCollector(env)
    warehouse = Warehouse(env, CONFIG, metrics)
    return env, metrics, warehouse

def test_warehouse_stations():
    env, metrics, warehouse = make_env()
    assert len(warehouse.stations) == CONFIG["num_stations"]

def test_travel_time_symmetry():
    env, metrics, warehouse = make_env()
    a, b =(0, 0), (5, 10)
    assert warehouse.travel_time(a, b) == warehouse.travel_time(b, a)

def test_travel_time_zero():
    env, metrics, warehouse = make_env()
    assert warehouse.travel_time((3,7), (3,7)) == 0.0

def test_task_id_increments():
    Task._task_id = 0
    t1 = Task((1, 1), 2.0, 0.0)
    t2 = Task((2, 2), 3.0, 1.0)
    assert t1.id == 1
    assert t2.id == 2

#---------integration tests--------------

def test_task_generator_creates_tasks():
    env, metrics, warehouse = make_env()
    task_gen = TaskGenerator(env, CONFIG, warehouse)
    env.process(task_gen.generate())
    env.run(until=10)
    assert len(warehouse.task_queue.items) > 0

def test_robot_completes_task():
    env, metrics, warehouse = make_env()
    task = Task(warehouse.stations[0], 2.0, 0.0)
    warehouse.task_queue.put(task)
    robot = Robot(env, 0, CONFIG, warehouse, metrics)
    env.process(robot.work())
    env.run(until=CONFIG["sim_duration"])
    assert len(metrics.tasks_completed) == 1

#---------system tests--------------

def test_full_sim_runs():
    from main import run_sim
    metrics = run_sim(CONFIG)
    assert len(metrics.tasks_completed) > 0
    assert metrics.throughput() > 0
