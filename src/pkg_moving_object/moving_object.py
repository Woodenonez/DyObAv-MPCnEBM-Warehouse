import copy
import math
import random
from typing import Union, Callable, Optional, Type, Sequence

import numpy as np
from shapely.geometry import Point, LineString # type: ignore
import matplotlib.patches as patches
from matplotlib.axes import Axes

from basic_motion_model.motion_model import MotionModel
from basic_motion_model.motion_model import HumanModel, OmnidirectionalModel
from basic_motion_model.motion_model import UnicycleModel


class MovingObject():
    def __init__(self, state: np.ndarray, ts: float, radius=1.0, stagger=0.0):
        """Moving object.
        
        Args:
            state: Initial state of the object, should be (x, y, theta)
            ts: Sampling time.
            radius: Radius of the object (assuming circular).
            stagger: Stagger of the object.

        Attributes:
            r: Radius of the object.
            ts: Sampling time.
            state: Current state of the object.
            stagger: Stagger of the object.
            motion_model: Motion model of the object. Default to OmnidirectionalModel.
            past_traj: List of tuples, past trajectory of the object.
            with_path: Whether the object has a path to follow.

        Notes:
            The motion model should be reloaded for different agents.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f'State must be numpy.ndarry, got {type(state)}.')
        self.r = radius
        self.ts = ts
        self.state = state
        self.stagger = stagger
        self.motion_model:MotionModel = OmnidirectionalModel(ts)
        self.past_traj = [tuple(self.state.tolist())]
        self.with_path = False

        self.sf_mode = False
        self.social_rep_opponent_type:Optional[Type['MovingObject']] = None

    @property
    def position(self) -> np.ndarray:
        return self.state[:2]
    
    @property
    def heading(self) -> float:
        return self.state[2]
    
    @property
    def velocity(self) -> np.ndarray:
        try:
            last_state_tuple = self.past_traj[-2]
        except IndexError:
            last_state_tuple = self.past_traj[-1]
        velocity = np.array([self.state[0] - last_state_tuple[0], self.state[1] - last_state_tuple[1]]) / self.ts
        return velocity
    
    @property
    def docking_point(self) -> tuple[float, float]:
        # choose the closest point on the path
        if not self.with_path:
            raise RuntimeError('Path is not set yet.')
        if not self.coming_path:
            return self.path[-1]
        
        state_point = Point(self.state[:2])
        docking_point:Point = self.path_shapely.interpolate(self.path_shapely.project(state_point))
        return docking_point.x, docking_point.y
        

    def set_path(self, path: list[tuple[float, float]]):
        self.with_path = True
        self.path = path
        self.coming_path = copy.deepcopy(path)
        self.past_traj = [tuple(self.state.tolist())]

        self.path_shapely = LineString(path)
    
    def set_social_repulsion(self, max_distance:float=5.0, max_angle:float=0.785, max_force:float=0.5, opponent_type:Optional[Type['MovingObject']]=None):
        """Set social repulsion (from other moving agents) parameters.

        Args:
            max_distance: Maximum distance for social repulsion to take effect.
            max_angle: Maximum angle in radian (between the agent's and the opponent's heading) that social repulsion can achieve.
            max_force: Maximum magnitude of the social repulsion force (velocity).
            opponent_type: Type of the opponent agent to consider for social repulsion, must be a subclass of MovingObject.

        Notes:
            If opponent_type is None (default), the agent will not consider social repulsion.
        """
        self.sf_mode = True
        self.social_rep_max_distance = abs(max_distance)
        self.social_rep_max_angle = abs(max_angle)
        self.social_rep_max_force = max_force
        self.social_rep_opponent_type = opponent_type

    def get_social_repulsion(self, agent_list: Sequence['MovingObject']) -> tuple[np.ndarray, list[np.ndarray], float]:
        """Get social repulsion force from other agents.

        Args:
            agent_list: A list of agents to consider for social repulsion.

        Returns:
            social_repulsion: The total social repulsion force.
            rep_forces: List of social repulsion forces from each opponent agent.
        """
        social_repulsion = np.array([0.0, 0.0])
        if (self.social_rep_opponent_type is None) or (not agent_list) or (not any(isinstance(agent, self.social_rep_opponent_type) for agent in agent_list)):
            self.social_repulsion = social_repulsion
            self.rep_forces = []
            return social_repulsion, [], 1.0
        
        rep_forces = []
        attenuation_factor = 1.0
        for agent in agent_list:
            if (agent == self) or (not isinstance(agent, self.social_rep_opponent_type)):
                continue

            relative_position = np.array([self.state[0] - agent.state[0], self.state[1] - agent.state[1]])
            relative_velocity = np.array([self.velocity[0] - agent.velocity[0], self.velocity[1] - agent.velocity[1]])
            if (np.linalg.norm(relative_position) > 0.5) and (np.linalg.norm(relative_velocity) != 0):
                if np.dot(relative_position/np.linalg.norm(relative_position), 
                        relative_velocity/np.linalg.norm(relative_velocity)) > -0.2: # not approaching
                    continue

            dist = math.hypot(agent.state[0] - self.state[0], agent.state[1] - self.state[1])
            if dist > self.social_rep_max_distance:
                continue
            rep_direction = np.array([self.state[0] - agent.state[0], self.state[1] - agent.state[1]])
            rep_direction = rep_direction / np.linalg.norm(rep_direction)
            rep_magnitude = self.social_rep_max_force * (1 - dist/self.social_rep_max_distance) * np.exp(-dist/self.social_rep_max_distance)
            rep_magnitude = min(1.5 * rep_magnitude, self.social_rep_max_force)
            rep_force = rep_direction * rep_magnitude
            rep_forces.append(rep_force)

            # attenuation_factor_new = 1.5 - (self.social_rep_max_distance - dist) / self.social_rep_max_distance
            # if attenuation_factor_new < attenuation_factor:
            #     attenuation_factor = attenuation_factor_new

        if rep_forces:
            desired_heading = math.atan2(self.coming_path[0][1] - self.state[1], self.coming_path[0][0] - self.state[0])
            social_repulsion = np.sum(rep_forces, axis=0)
            if social_repulsion[0] == 0 and social_repulsion[1] == 0: # balanced forces
                return social_repulsion, [], 1.0
            total_rep_magnitude = np.linalg.norm(social_repulsion)
            if total_rep_magnitude > self.social_rep_max_force: # cap the force
                social_repulsion = social_repulsion / total_rep_magnitude * self.social_rep_max_force
            total_rep_angle = math.atan2(social_repulsion[1], social_repulsion[0])
            check_heading = (self.heading + desired_heading*4) / 5
            if check_heading - total_rep_angle > self.social_rep_max_angle:
                if check_heading - total_rep_angle > math.pi:
                    total_rep_direction = np.array([math.cos(check_heading+self.social_rep_max_angle), 
                                                    math.sin(check_heading+self.social_rep_max_angle)])
                else:
                    total_rep_direction = np.array([math.cos(check_heading-self.social_rep_max_angle), 
                                                    math.sin(check_heading-self.social_rep_max_angle)])
                social_repulsion = total_rep_direction * np.linalg.norm(social_repulsion)
            elif check_heading - total_rep_angle < -self.social_rep_max_angle:
                if check_heading - total_rep_angle < -math.pi:
                    total_rep_direction = np.array([math.cos(check_heading-self.social_rep_max_angle), 
                                                    math.sin(check_heading-self.social_rep_max_angle)])
                else:
                    total_rep_direction = np.array([math.cos(check_heading+self.social_rep_max_angle), 
                                                    math.sin(check_heading+self.social_rep_max_angle)])
                social_repulsion = total_rep_direction * np.linalg.norm(social_repulsion)

        self.social_repulsion = social_repulsion
        self.rep_forces = rep_forces

        return social_repulsion, rep_forces, attenuation_factor


    def get_next_goal(self, vmax: float) -> Union[tuple, None]:
        if not self.with_path:
            raise RuntimeError('Path is not set yet.')
        if not self.coming_path:
            return None
        if len(self.coming_path) > 1:
            dist_to_next_goal = math.hypot(self.coming_path[0][0] - self.docking_point[0], self.coming_path[0][1] - self.docking_point[1])
        else:
            dist_to_next_goal = math.hypot(self.coming_path[0][0] - self.state[0], self.coming_path[0][1] - self.state[1])
        if self.sf_mode and np.linalg.norm(self.social_repulsion) > 0:
            if dist_to_next_goal < (vmax*self.ts*2):
                self.coming_path.pop(0)
        else:
            if dist_to_next_goal < (vmax*self.ts):
                self.coming_path.pop(0)
        if self.coming_path:
            return self.coming_path[0]
        else:
            return None

    def get_action(self, next_path_node: tuple, vmax: float) -> np.ndarray:
        stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
        dist_to_next_node = math.hypot(self.coming_path[0][0] - self.state[0], self.coming_path[0][1] - self.state[1])
        dire = ((next_path_node[0] - self.state[0])/dist_to_next_node, 
                (next_path_node[1] - self.state[1])/dist_to_next_node)
        action:np.ndarray = np.array([dire[0]*vmax+stagger, dire[1]*vmax+stagger])
        return action

    def one_step(self, action: np.ndarray):
        self.state = self.motion_model(self.state, action)
        self.past_traj.append(tuple(self.state.tolist()))

    def run_step(self, vmax: float, social_force:Optional[np.ndarray]=None, attenuation_factor:float=1.0) -> Optional[np.ndarray]:
        """Run the agent one step along the path (need to be preset) with optional social force.

        Args:
            vmax: Maximum velocity of the agent.
            social_force: The total social force. Defaults to None.

        Returns:
            action: The action taken by the agent.

        Notes:
            Use `one_step` method if no path is set.
        """
        next_path_node = self.get_next_goal(vmax)
        if next_path_node is None:
            return None
        action = self.get_action(next_path_node, vmax)
        if social_force is not None:
            action[0] += social_force[0]
            action[1] += social_force[1]
        action = action * attenuation_factor
        self.one_step(action)
        return action

    def run(self, path: list[tuple[float, float]], vmax:float=0.5):
        """Run the agent along the path until the end.

        Note:
            To run the agent step by step, use `run_step` method.
        """
        self.set_path(path)
        done = False
        while (not done):
            action = self.run_step(vmax)
            done = (action is None)

    def plot_agent(self, ax:Axes, color:str='b', ct:Optional[Callable]=None):
        if ct is not None:
            robot_patch = patches.Circle(ct(self.state[:2]), self.r, color=color)
        else:
            robot_patch = patches.Circle(self.state[:2], self.r, color=color) # type: ignore
        ax.add_patch(robot_patch)

    def plot_social_force(self, ax:Axes, color:str='r', length_inverse_scale:float=1.0, plot_all:bool=False):
        ax.quiver(self.state[0], self.state[1], self.social_repulsion[0], self.social_repulsion[1], angles='xy', scale_units='xy', scale=length_inverse_scale, color=color)
        if plot_all:
            for rep_force in self.rep_forces:
                ax.quiver(self.state[0], self.state[1], rep_force[0], rep_force[1], angles='xy', scale_units='xy', scale=length_inverse_scale, color=color, alpha=0.5)


class HumanObject(MovingObject):
    def __init__(self, state: np.ndarray, ts: float, radius: float, stagger: float):
        super().__init__(state, ts, radius, stagger)
        self.motion_model = HumanModel(self.ts)


class RobotObject(MovingObject):
    def __init__(self, state: np.ndarray, ts: float, radius: float):
        super().__init__(state, ts, radius, 0)
        self.motion_model = UnicycleModel(self.ts, rk4=True)
