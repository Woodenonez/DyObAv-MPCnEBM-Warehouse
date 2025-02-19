import sys
import math
from timeit import default_timer as timer
from copy import copy, deepcopy
from typing import Optional, Union, Type, cast
from scipy import interpolate


import numpy as np
import matplotlib.pyplot as plt
import casadi as ca # type: ignore

from matplotlib.axes import Axes


def downsample_ref_states(original_states: np.ndarray, original_speed: float, new_speed: float):
    """Downsample the reference states to the given speed. 

    Args:
        original_states: The original reference states, each row is a state.
        original_speed: The original speed used to generate the original reference states.
        new_speed: The new speed to downsample the reference states.

    Returns:
        New reference states with the given speed, each row is a state.

    Notes:
        The new speed should be smaller than the original speed. Otherwise, the will be few states in the new reference.
    """
    n_states = original_states.shape[0]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(original_states, axis=0)**2, axis=1))) # distance traveled along the path at each point
    distances = np.insert(distances, 0, 0)/distances[-1] # normalize distances to [0, 1]
    fx = interpolate.interp1d(distances, original_states[:, 0], kind='linear')
    fy = interpolate.interp1d(distances, original_states[:, 1], kind='linear')

    num_points = int(original_speed/new_speed*n_states)  
    new_distances = np.linspace(0, 1, num_points)
    new_x = fx(new_distances)
    new_y = fy(new_distances)
    new_heading = np.arctan2(np.diff(new_y), np.diff(new_x))
    new_heading = np.append(new_heading, new_heading[-1])
    new_states = np.column_stack([new_x, new_y, new_heading])[:n_states, :]
    return new_states


original_states = np.array(
      [[4.66000051, 2.3       , 3.14159265],
       [4.42000054, 2.3       , 3.14159265],
       [4.18000058, 2.3       , 3.14159265],
       [3.94000061, 2.3       , 3.14159265],
       [3.70000064, 2.3       , 3.14159265],
       [3.46000067, 2.3       , 3.14159265],
       [3.2200007 , 2.3       , 3.14159265],
       [2.98000074, 2.3       , 3.14159265],
       [2.74000077, 2.3       , 3.14159265],
       [2.5000008 , 2.3       , 3.14159265],
       [2.26000083, 2.3       , 3.14159265],
       [2.02000086, 2.3       , 3.14159265],
       [1.7800009 , 2.3       , 3.14159265],
       [1.54000093, 2.3       , 3.14159265],
       [1.30000096, 2.3       , 3.14159265],
       [1.06000099, 2.3       , 3.14159265],
       [1.        , 2.47999898, 1.57079633],
       [1.        , 2.71999896, 1.57079633],
       [1.        , 2.95999893, 1.57079633],
       [1.        , 3.19999891, 1.57079633]])
original_speed = 1.2
new_speed = 0.96

# new_states = downsample_ref_states(original_states, original_speed, new_speed)

n_states = original_states.shape[0]
distances = np.cumsum(np.sqrt(np.sum(np.diff(original_states[:, :2], axis=0)**2, axis=1))) # distance traveled along the path at each point
distances = np.insert(distances, 0, 0)/distances[-1] # normalize distances to [0, 1]
fx = interpolate.interp1d(distances, original_states[:, 0], kind='linear')
fy = interpolate.interp1d(distances, original_states[:, 1], kind='linear')

num_points = int(original_speed/new_speed*n_states)  
new_distances = np.linspace(0, 1, num_points)
new_x = fx(new_distances)
new_y = fy(new_distances)
new_states = np.column_stack([new_x, new_y])

fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
ax1.plot(original_states[:, 0], original_states[:, 1], 'x', label='Original')
ax1.plot(new_states[:, 0], new_states[:, 1],'o', markerfacecolor='none', label='Downsampled')
ax1.set_aspect('equal')
ax1.legend()

ax2.plot(distances, original_states[:, 0], 'x', label='Original')
ax2.plot(new_distances, new_x, 'o', markerfacecolor='none', label='Downsampled')
ax2.set_aspect('equal')
ax2.legend()

ax3.plot(distances, original_states[:, 1], 'x', label='Original')
ax3.plot(new_distances, new_y, 'o', markerfacecolor='none', label='Downsampled')
ax3.set_aspect('equal')
ax3.legend()

plt.show()