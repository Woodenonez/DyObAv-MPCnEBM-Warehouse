import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pkg_moving_object.moving_object import MovingObject, HumanObject, RobotObject

from typing import cast, Union

scale = 0.2


human_1 = HumanObject(np.array([0.0, 0.0, 0.0]), 0.1, 0.2, 0.1)
human_1_path = [(0.0, 0.0), (6.0, 0.0), (3.0, 3.0)]
human_2 = HumanObject(np.array([3.0, 3.0, 0.0]), 0.1, 0.2, 0.1)
human_2_path = [(3.0, 3.0), (6.0, 1.5), (0.0, 0.0)]

robot_1 = RobotObject(np.array([3.0, 0.0, 0.0]), 0.1, 0.5)
robot_2 = RobotObject(np.array([4.0, 2.0, 0.0]), 0.1, 0.5)

humans = [human_1, human_2]
hpaths = [human_1_path, human_2_path]
robots = [robot_1, robot_2]
agents = humans + robots

for human, hpath in zip(humans, hpaths):
    human.set_path(hpath)
    human.set_social_repulsion(max_distance=5.0, max_angle=math.radians(90), max_force=0.5, opponent_type=RobotObject)

fig, ax = plt.subplots()
ax = cast(Axes, ax)
ax.axis('equal')
while 1:
    ax.clear()
    for human in humans:
        social_force, rep_forces = human.get_social_repulsion(agents)
        action = human.run_step(vmax=0.5, social_force=social_force)
        ax.plot(np.array(human.path)[:, 0], np.array(human.path)[:, 1], 'kx--')
        ax.plot(np.array(human.past_traj)[:, 0], np.array(human.past_traj)[:, 1], 'b.')
        if action is not None:
            ax.quiver(human.state[0], human.state[1], action[0]-social_force[0], action[1]-social_force[1], angles='xy', scale_units='xy', scale=scale, color='gray')
            ax.quiver(human.state[0], human.state[1], action[0], action[1], angles='xy', scale_units='xy', scale=scale, color='k')
        human.plot_agent(ax, color='b')
        human.plot_social_force(ax, color='b', plot_all=True, length_inverse_scale=scale)
    for robot in robots:
        robot.plot_agent(ax, color='r')
    
    ax.set_xlim(-5, 8)
    ax.set_ylim(-5, 8)
    plt.draw()
    plt.pause(0.01)
    while not plt.waitforbuttonpress():
        pass

plt.show()