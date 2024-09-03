from abc import ABC, abstractmethod
from enum import Enum
import math
from copy import deepcopy
from typing import Union, Callable, Optional

import numpy as np
import casadi.casadi as cs # type: ignore

"""
A motion model is either holonomic or non-holonomic.
A holonomic model is a model where the change of the state is a function of the action only.
A non-holonomic model is a model where the change of the state is a function of the action and the state.

Supported holonomic (additive) models:
    - Omnidirectional model
        - Human model (auto-tuned)
Supported non-holonomic models:
    - Unicycle model (differential drive)
Ongoing:
    - Simple car model (non-holonomic)
Wishlist:
    - Ackermann model (non-holonomic)

Notes:
    - The state is always [x, y, theta] in this module.
    - The action is [a, b, c, ...] in this module.
    - The sampling time is always `ts` in this module.
    - The motion model is a function `f(s, a, ts)`.
    - The motion model can be called with `f(s, a, ts)`.
"""


class MotionModelType(Enum):
    """The type of a motion model."""
    HOLONOMIC = 0
    NON_HOLONOMIC = 1
    PRESET = 2
    UNKNOWN = 3

class MotionModel(ABC):
    """An interface for a motion model under `numpy`.
    `next_s = f(s, a, ts)`
    
    Properties:
        motion_model_type: The type of the motion model.
        
    Methods:
        __call__: The motion model.
        zero_state: Return the zero state of the motion model.
        zero_action: Return the zero action of the motion model.
    """
    _motion_model_type: MotionModelType

    def __init__(self, model: Callable, state_dim: int, action_dim: int, sampling_time: float) -> None:
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ts = sampling_time

    def __call__(self, state: np.ndarray, action: np.ndarray, ts:Optional[float]=None, **kwargs) -> np.ndarray:
        """The sampling time can be changed at runtime."""
        if isinstance(ts, float) or isinstance(ts, int):
            self.ts = float(ts)
        return self.model(state, action, self.ts, **kwargs)

    @property
    def motion_model_type(self) -> MotionModelType:
        return self._motion_model_type

    def zero_state(self, casadi=False) -> Union[np.ndarray, cs.SX]:
        """Return the zero state of the motion model."""
        if casadi:
            return cs.DM.zeros(self.state_dim)
        return np.zeros(self.state_dim)

    def zero_action(self, casadi=False) -> Union[np.ndarray, cs.SX]:
        """Return the zero action of the motion model."""
        if casadi:
            return cs.DM.zeros(self.action_dim)
        return np.zeros(self.action_dim)



class OmnidirectionalModel(MotionModel):
    """Omnidirectional model.

    Args:
        state: x, y, and theta.
        action: velocity (x and y) and angular speed.
    """
    _motion_model_type = MotionModelType.HOLONOMIC

    def __init__(self, sampling_time: float) -> None:
        super().__init__(omnidirectional_model, 3, 3, sampling_time)


class HumanModel(MotionModel):
    """Human model is based on the omnidirectional model.

    Args:
        state: x, y, and theta.
        action: velocity (vx and vy).
    """
    _motion_model_type = MotionModelType.HOLONOMIC

    def __init__(self, sampling_time: float) -> None:
        super().__init__(human_model, 3, 2, sampling_time)


class UnicycleModel(MotionModel):
    """Unicycle model.

    Args:
        state: x, y, and theta.
        action: speed and angular speed.
    """
    _motion_model_type = MotionModelType.NON_HOLONOMIC

    def __init__(self, sampling_time: float, rk4:bool=True) -> None:
        super().__init__(unicycle_model, 3, 2, sampling_time)
        self.rk4 = rk4

    def __call__(self, state: np.ndarray, action: np.ndarray, ts:Optional[float]=None, **kwargs) -> np.ndarray:
        return super().__call__(state, action, ts, rk4=self.rk4)


class ReciprocatingModel(MotionModel):
    """Reciprocating model under `numpy`.

    Args:
        state: x, y, and theta.
        action: speed and angular speed.
    """
    _motion_model_type = MotionModelType.PRESET

    def __init__(self, sampling_time: float, p1: tuple, p2: tuple) -> None:
        """
        Args:
            p1: The start point.
            p2: The other point.
        """
        super().__init__(reciprocating_model, 3, 1, sampling_time)
        self.p1 = p1
        self.p2 = p2

    def __call__(self, state: np.ndarray, action: np.ndarray, ts:Optional[float]=None, *, kt=None, **kwargs) -> np.ndarray:
        """kt: The current time step. Run `quick_call` instead"""
        if kt is None:
            raise ValueError("The current time step should be provided.")
        return super().__call__(state, action, ts, kt=kt, p1=self.p1, p2=self.p2)
    
    def quick_call(self, kt: int, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.__call__(state, action, kt=kt)

    def init_state(self):
        return np.array([self.p1[0], self.p1[1], 0])



def omnidirectional_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float) -> Union[np.ndarray, cs.SX]:
    """Omnidirectional model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: velocity (x and y) and angular speed.
    """
    if action.shape[0] != 3:
        raise ValueError("The action should be [vx, vy, omega].")
    new_state = state + ts * action
    return new_state

def human_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float) -> Union[np.ndarray, cs.SX]:
    """Human model is based on the omnidirectional model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: velocity (vx and vy).

    Notes:
        The human model is auto-tuned, which means the heading is always towards the velocity direction.
    """
    new_state = deepcopy(state)
    new_state[0] += ts * action[0]
    new_state[1] += ts * action[1]
    new_state[2] = math.atan2(action[1], action[0])
    return new_state

def unicycle_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float, rk4:bool=True) -> Union[np.ndarray, cs.SX]:
    """Unicycle model.
    
    Args:
        ts: Sampling time.
        state: x, y, and theta.
        action: speed and angular speed.
        rk4: If True, use Runge-Kutta 4 to refine the model.
    """
    def d_state_f(state, action):
        if isinstance(state, cs.SX):
            return ts * cs.vertcat(action[0]*cs.cos(state[2]), action[0]*cs.sin(state[2]), action[1])
        return ts * np.array([action[0]*np.cos(state[2]), action[0]*np.sin(state[2]), action[1]])
    if rk4:
        k1 = d_state_f(state, action)
        k2 = d_state_f(state + 0.5*k1, action)
        k3 = d_state_f(state + 0.5*k2, action)
        k4 = d_state_f(state + k3, action)
        d_state = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        d_state = d_state(state, action)

    return state + d_state

def reciprocating_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float, kt: int, p1: tuple, p2: tuple) -> Union[np.ndarray, cs.SX]:
    """Reciprocating model (start from p1).
    
    Args:
        ts: Sampling time.
        state: x, y, theta.
        action: speed.
        kt: The current time step.
        p1: The first point.
        p2: The second point.
    """
    if action[0] == 0:
        return state
    
    discrete_period = int(2 * np.linalg.norm(np.array(p1) - np.array(p2)) / action[0] / ts) + 1
    progress = kt % discrete_period / discrete_period
    if progress < 0.5:
        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    else:
        theta = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    x = 2*abs(0.5-progress) * p1[0] + 2*(0.5-abs(0.5-progress)) * p2[0]
    y = 2*abs(0.5-progress) * p1[1] + 2*(0.5-abs(0.5-progress)) * p2[1]
    if isinstance(state, cs.SX):
        return cs.vertcat(x, y, theta)
    return np.array([x, y, theta])

def car_model(state: Union[np.ndarray, cs.SX], action: Union[np.ndarray, cs.SX], ts: float) -> Union[np.ndarray, cs.SX]:
    """http://msl.cs.uiuc.edu/planning/node658.html"""
    raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = ReciprocatingModel(0.1, (0, 0), (1, 1))
    state = model.init_state()

    fig, ax = plt.subplots()
    for i in range(100):
        ax.cla()
        ax.plot([0, 1], [0, 1], 'k--')
        state = model.quick_call(i, state, np.array([0.5]))
        ax.plot(state[0], state[1], 'ro')
        ax.axis('equal')
        plt.pause(0.1)

    plt.show()