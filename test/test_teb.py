import os
import pathlib
from timeit import default_timer as timer
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from configs import TebConfiguration, CircularRobotSpecification
from pkg_planner_teb.trajectory_planner import TrajectoryPlanner


p_start = np.array([0.0, 0.0, 0.0])
p_goal = np.array([10.0, 10.0, 0.0])
obstacles = np.array([
    [7.0, 7.0, 1.0],
    [3.0, 3.0, 1.0],
])
obstacle_x = obstacles[:, 0]
obstacle_y = obstacles[:, 1]
obstacle_r = obstacles[:, 2]

K = 100 # simulation steps
N = 20
v_max = 1.0

x_init = np.linspace(p_start[0], p_goal[0], N)
y_init = np.linspace(p_start[1], p_goal[1], N)
dt_init = np.ones(N-1) * (np.linalg.norm(p_goal[:2] - p_start[:2]) / (N-1) / v_max)

project_dir = pathlib.Path(__file__).resolve().parents[1]
cfg_tracker_path = os.path.join(project_dir, 'config', 'teb.yaml')
cfg_robot_path = os.path.join(project_dir, 'config', 'spec_robot.yaml')

config = TebConfiguration.from_yaml(cfg_tracker_path)
robot_spec = CircularRobotSpecification.from_yaml(cfg_robot_path)

ref_states = np.vstack((x_init, y_init)).T
planner = TrajectoryPlanner(config, robot_spec, safe_factor=2.0, verbose=False)
planner.set_ref_states(p_start, ref_states, ref_speed=v_max)


fig, ax = plt.subplots()
ax = cast(Axes, ax)
for k in range(K):

    obstacles[0, 0] += -np.cos(k*np.pi/10) * 0.5
    obstacles[0, 1] +=  np.cos(k*np.pi/10) * 0.5
    obstacles[1, 0] +=  np.cos(k*np.pi/10) * 0.5
    obstacles[1, 1] += -np.cos(k*np.pi/10) * 0.5
    desired_states, debug_info = planner.run_step(obstacles[:, :2].tolist(), obstacles[:, 2].tolist())
    
    ax.cla()
    ax.plot(x_init, y_init, '-x', label='Initial Trajectory')
    ax.plot(desired_states[:, 0], desired_states[:, 1], '-o', label='Optimized Trajectory')
    ax.plot(p_start[0], p_start[1], 'go', label='Start')
    ax.plot(p_goal[0], p_goal[1], 'ro', label='Goal')
    ax.plot(obstacles[:, 0], obstacles[:, 1], 'ko', label='Obstacles')
    for i in range(obstacles.shape[0]):
        circle = plt.Circle((obstacles[i, 0], obstacles[i, 1]), obstacles[i, 2], color='k', fill=False)
        ax.add_artist(circle)

    ax.axis('equal')
    ax.grid(True)

    plt.pause(0.2)
    # while not plt.waitforbuttonpress():
    #     pass

plt.show()