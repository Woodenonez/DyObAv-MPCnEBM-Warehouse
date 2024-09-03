from typing import Any, Optional, Union

import numpy as np

from basic_motion_model import motion_model as momo
from pkg_moving_object.moving_object import RobotObject
from configs import CircularRobotSpecification, MpcConfiguration

# Type hinting
from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker as TrajectoryTrackerMPC
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from visualizer.object import ObjectVisualizer


TrackerType = Union[TrajectoryTrackerMPC]


MAX_NUMBER_OF_ROBOTS = 10


class Robot:
    """Robot is a wrapper class for a robot object."""

    _id_list = [-1]

    def __init__(self, config: CircularRobotSpecification, motion_model: momo.MotionModel, id_:Optional[int]=None, name:Optional[str]=None) -> None:
        self.config = config
        self.motion_model = motion_model
        self._check_identifier(id_, name)
        self._priority = 0

    def _check_identifier(self, id_: Optional[int], name: Optional[str]) -> None:
        if id_ is None:
            if max(self._id_list) > MAX_NUMBER_OF_ROBOTS:
                raise ValueError('Maximum number of robots reached.')
            id_ = max(self._id_list)+1 if self._id_list else 0
        elif id_ in self._id_list:
            raise ValueError(f'A robot with id {id_} already exists.')
        self._id = id_
        self._id_list.append(id_)
        if name is None:
            name = f'{self.__class__.__name__}_{id_}'
        self._name = name
        
    @property
    def id_(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def priority(self) -> int:
        return self._priority
    
    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value
    
    @property
    def state(self) -> np.ndarray:
        self._state[2] = self._state[2] % (2*np.pi)
        return self._state
    
    @property
    def past_traj(self) -> list[tuple]:
        return self.robot_object.past_traj

    def set_state(self, state: np.ndarray) -> None:
        self._state = state
        self.robot_object = RobotObject(state=state, ts=self.config.ts, radius=self.config.vehicle_width/2)
        self.robot_object.motion_model = self.motion_model

    def step(self, action: np.ndarray) -> None:
        self.robot_object.one_step(action)
        self._state = self.robot_object.state

    @staticmethod
    def reset_id_list() -> None:
        Robot._id_list = [-1]


class RobotUnit():
    """A robot unit is a dictionary-like object that contains a robot, a controller, a visualizer, and a reference path."""
    def __init__(self, robot: Robot, controller: TrackerType, planner: LocalTrajPlanner, visualizer: ObjectVisualizer) -> None:
        self.robot = robot
        self.controller = controller
        self.planner = planner
        self.visualizer = visualizer

        self.start = None
        self.goal = None
        self.idle = True
        self.pred_states = None

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


class RobotManager():
    """Manager for robots.

    Notes:
        To access a robot unit (containing a robot and other attributes):
    - `robot_manager(robot_id).XX`: return a component XX
    - `robot_manager.get_XX(robot_id)`: return XX
    """

    ROBOT_ID_LIST:list[int] = []

    def __init__(self) -> None:
        self.reset_id_list()
        self._robot_dict:dict[Any, RobotUnit] = {}
    
    def __call__(self, robot_id) -> RobotUnit:
        return self._robot_dict[robot_id]
    
    def __len__(self) -> int:
        return len(self.ROBOT_ID_LIST)
    
    @staticmethod
    def _check_id(f): 
        """Decorator to check if robot_id exists"""
        def wrapper(self, robot_id, *args, **kwargs):
            if robot_id not in self.ROBOT_ID_LIST:
                raise ValueError(f'Robot {robot_id} does not exist!')
            return f(self, robot_id, *args, **kwargs)
        return wrapper
    
    def reset_id_list(self) -> None:
        Robot.reset_id_list()
        self.ROBOT_ID_LIST = []
    
    ### Basic operations
    def create_robot(self, config: CircularRobotSpecification, motion_model: Optional[momo.MotionModel], id_:Optional[int]=None, name:Optional[str]=None) -> Robot:
        """Create a robot object.

        Notes:
            Motion model: If not provided, a unicycle model is used.
            ID and name: It is suggested to provide an ID at least.
        """
        if motion_model is None:
            motion_model = momo.UnicycleModel(sampling_time=config.ts)
        robot = Robot(config, motion_model, id_, name)
        return robot

    def add_robot(self, robot: Robot, controller: TrackerType, planner: LocalTrajPlanner, visualizer: ObjectVisualizer) -> None:
        if robot.id_ in self.ROBOT_ID_LIST:
            raise ValueError(f'Robot {robot.id_} exists! Cannot add it again.')
        self._robot_dict[robot.id_] = RobotUnit(robot, controller, planner, visualizer)
        self.ROBOT_ID_LIST.append(robot.id_)

    @_check_id
    def add_schedule(self, robot_id, current_state: np.ndarray, scheduled_path_coords: list[tuple[float, float]], scheduled_path_times: Optional[list[float]]=None):
        """Add a schedule for a robot in the manager.

        Arguments:
            robot_id: robot id

        Notes:
            The start, goal states and idle flags are set.
        """
        if robot_id not in self.ROBOT_ID_LIST:
            raise ValueError(f'[{self.__class__.__name__}] Robot {robot_id} does not exist!')
        if self.get_robot_idle(robot_id) is False:
            raise ValueError(f'[{self.__class__.__name__}] Robot {robot_id} is not idle!')
        
        planner = self.get_planner(robot_id)
        planner.load_path(scheduled_path_coords, scheduled_path_times, nomial_speed=self.get_robot(robot_id).config.lin_vel_max, method="linear")

        # start_coord = graph.get_node_coord(path_nodes[0])
        # start_coord_next = graph.get_node_coord(path_nodes[1])
        # start_heading = np.arctan2(start_coord_next[1]-start_coord[1], start_coord_next[0]-start_coord[0])
        goal_coord = scheduled_path_coords[-1]
        goal_coord_prev = scheduled_path_coords[-2]
        goal_heading = np.arctan2(goal_coord[1]-goal_coord_prev[1], goal_coord[0]-goal_coord_prev[0])
        goal_state = np.array([*goal_coord, goal_heading])

        self.set_start_state(robot_id, current_state)
        self.set_goal_state(robot_id, goal_state)
        self.set_robot_idle(robot_id, False)

        controller = self.get_controller(robot_id)
        controller.load_init_states(current_state, goal_state)

    @_check_id
    def remove_robot(self, robot_id) -> None:
        self._robot_dict.pop(robot_id)
        self.ROBOT_ID_LIST.remove(robot_id)

    @_check_id
    def set_robot_unit(self, robot_id, **kwargs) -> None:
        """Set robot attributes.
        
        Possible attributes are:
        - `start`: start state
        - `goal`: goal state
        - `planner`: motion planner
        - `controller`: controller
        - `visualizer`: visualizer
        - `idle`: idle flag
        - `pred_states`: predicted states
        """
        for key, value in kwargs.items():
            self._robot_dict[robot_id][key] = value

    @_check_id
    def get_robot_unit(self, robot_id) -> RobotUnit:
        return self._robot_dict[robot_id]

    ### Get/set robot attributes
    def get_robot(self, robot_id) -> Robot:
        return self.get_robot_unit(robot_id).robot
    
    def get_all_robots(self) -> list[Robot]:
        return [self.get_robot(robot_id) for robot_id in self.ROBOT_ID_LIST]
    
    def get_start_state(self, robot_id) -> np.ndarray:
        return self.get_robot_unit(robot_id).start
    
    def set_start_state(self, robot_id, state: np.ndarray) -> None:
        self.set_robot_unit(robot_id, start=state)
    
    def get_goal_state(self, robot_id) -> np.ndarray:
        return self.get_robot_unit(robot_id).goal
    
    def set_goal_state(self, robot_id, state: np.ndarray) -> None:
        self.set_robot_unit(robot_id, goal=state)
    
    def get_robot_idle(self, robot_id) -> bool:
        return self.get_robot_unit(robot_id).idle
    
    def set_robot_idle(self, robot_id, idle: bool) -> None:
        self.set_robot_unit(robot_id, idle=idle)

    def get_pred_states(self, robot_id) -> np.ndarray:
        return self.get_robot_unit(robot_id).pred_states
    
    def set_pred_states(self, robot_id, pred_states: np.ndarray) -> None:
        self.set_robot_unit(robot_id, pred_states=pred_states)

    ### Check controller, planner, and visualizer
    def get_controller(self, robot_id) -> TrackerType:
        return self.get_robot_unit(robot_id).controller
    
    def set_controller(self, robot_id, controller) -> None:
        self.set_robot_unit(robot_id, controller=controller)

    def get_planner(self, robot_id) -> LocalTrajPlanner:
        return self.get_robot_unit(robot_id).planner
    
    def set_planner(self, robot_id, planner: LocalTrajPlanner) -> None:
        self.set_robot_unit(robot_id, planner=planner)
    
    def get_visualizer(self, robot_id):
        return self.get_robot_unit(robot_id).visualizer
    
    def set_visualizer(self, robot_id, visualizer) -> None:
        self.set_robot_unit(robot_id, visualizer=visualizer)

    ### Check robot states
    def set_robot_state(self, robot_id, state: np.ndarray) -> None:
        robot = self.get_robot(robot_id)
        robot.set_state(state)

    def get_robot_state(self, robot_id) -> np.ndarray:
        robot = self.get_robot(robot_id)
        return robot.state
        
    @_check_id
    def get_other_robot_states(self, ego_robot_id, config_mpc: MpcConfiguration, default:float=-10.0) -> list:
        state_dim = config_mpc.ns
        horizon = config_mpc.N_hor
        num_others = config_mpc.Nother
        
        other_robot_states = [default] * state_dim * (horizon+1) * num_others
        idx = 0
        idx_pred = state_dim * num_others
        for id_ in list(self._robot_dict):
            if id_ != ego_robot_id:
                current_state:np.ndarray = self.get_robot_state(id_)
                pred_states:np.ndarray = self.get_pred_states(id_) # every row is a state
                other_robot_states[idx : idx+state_dim] = list(current_state)
                idx += state_dim
                if pred_states is not None:
                    other_robot_states[idx_pred : idx_pred+state_dim*horizon] = list(pred_states.reshape(-1))
                    idx_pred += state_dim*horizon
        return other_robot_states
    