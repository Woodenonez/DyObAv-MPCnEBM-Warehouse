from abc import ABC, abstractmethod
from enum import Enum

from typing import Any, Callable


MAX_NUMBER_OF_OBSTACLES = 999


class ObstacleMotionType(Enum):
    """The motion type of an obstacle."""
    STATIC = 0
    DYNAMIC = 1
    UNKNOWN = 2

class ObstacleShape(Enum):
    """The shape of an obstacle."""
    CIRCLE = 0
    ELLIPSE = 1
    POLYGON = 2


class ObstacleBase(ABC):
    """An abstract obstacle class.

    Attributes:
        geometry: The geometry of the obstacle.
        motion_model: The motion model of the obstacle, `s'=f(s, u, (ts))`.

    Properties:
        id_: The id of the obstacle.
        name: The name of the obstacle.
        obstacle_shape_type: The shape type of the obstacle.
        obstacle_motion_type: The motion type of the obstacle.
    """
    _id_list = [-1]

    def __init__(self, geometry, geometry_shape: str, motion_model:Callable=None, id_:int=None, name:str=None) -> None:
        self._check_identifier(id_, name)
        self._geometry = geometry
        self._motion_model = motion_model
        self._obstacle_shape_type = self._get_shape_type(geometry_shape)
        self._obstacle_motion_type = ObstacleMotionType.STATIC if motion_model is None else ObstacleMotionType.DYNAMIC

    def __str__(self) -> str:
        """Should not be overwritten."""
        return f'{self.__class__.__name__} [{self.obstacle_motion_type}] ID {self.id_}, name {self.name}'

    def _check_identifier(self, id_: int, name: str) -> None:
        if id_ is None:
            if max(self._id_list) > MAX_NUMBER_OF_OBSTACLES:
                raise ValueError('Maximum number of obstacles reached.')
            id_ = max(self._id_list)+1 if self._id_list else 0
        elif id_ < 0:
            raise ValueError('The id of an obstacle must be positive.')
        elif id_ in self._id_list:
            raise ValueError(f'An obstacle with id {id_} already exists.')
        self._id = id_
        self._id_list.append(id_)
        if name is None:
            name = f'{self.__class__.__name__}_{id_}'
        self._name = name

    def _get_shape_type(self, geometry_shape: str) -> ObstacleShape:
        if geometry_shape.lower() == 'circle':
            return ObstacleShape.CIRCLE
        elif geometry_shape.lower() == 'ellipse':
            return ObstacleShape.ELLIPSE
        elif geometry_shape.lower() == 'polygon':
            return ObstacleShape.POLYGON
        else:
            raise ValueError(f'Unknown obstacle shape {geometry_shape}.')

    @property
    def id_(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> tuple:
        pass

    @property
    @abstractmethod
    def state(self) -> tuple:
        """Define the state of the obstacle."""
        pass

    @property
    def obstacle_motion_type(self) -> ObstacleMotionType:
        return self._obstacle_motion_type

    @property
    def obstacle_shape_type(self) -> ObstacleShape:
        return self._obstacle_shape_type

    @property
    def geometry(self) -> Any:
        return self._geometry

    @property
    def motion_model(self) -> Callable:
        return self._motion_model
    
    @motion_model.setter
    def motion_model(self, motion_model:Callable=None) -> None:
        self._motion_model = motion_model
        self._obstacle_motion_type = ObstacleMotionType.STATIC if motion_model is None else ObstacleMotionType.DYNAMIC 

    def step(self, action: Any, dt:float=None) -> None:
        """Should be overwritten, if the motion model is not `None`."""
        pass


if __name__ == "__main__":
    o = Obstacle(None)
    