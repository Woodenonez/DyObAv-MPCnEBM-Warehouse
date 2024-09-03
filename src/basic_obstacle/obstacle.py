import numpy as np

from ._obstacle import ObstacleBase
from .geometry_plain import PlainGeometry
from .geometry_plain import PlainPolygon
from .geometry_plain import PlainCircle
from .geometry_plain import PlainEllipse

from .geometry_tools import GeometryTools

from typing import Callable, Any, Tuple, List
from matplotlib.axes import Axes

"""
All obstacles are based on the PlainGeometry objects.
"""


class Obstacle(ObstacleBase):
    def __init__(self, geometry: PlainGeometry, geometry_shape: str, motion_model:Callable=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, geometry_shape, motion_model, id_, name)
        self.geometry: PlainGeometry
        self.inflated_geometry: PlainGeometry = None

        self.geo_tools = GeometryTools()

    def __call__(self):
        raise NotImplementedError

    @classmethod
    def from_raw(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def state(self) -> np.ndarray:
        raise NotImplementedError
    
    def plot(self, ax: Axes, **kwargs) -> None:
        pass


class PolygonObstacle(Obstacle):
    def __init__(self, geometry: PlainPolygon, motion_model:Callable=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "polygon", motion_model, id_, name)
        self.geometry: PlainPolygon
        self.inflated_geometry: PlainPolygon = None

    def __call__(self, inflated=False) -> List[tuple]:
        if inflated:
            return self.inflated_geometry()
        return self.geometry()
    
    @classmethod
    def from_raw(cls, vertices:List[tuple], motion_model:Callable=None, id_:int=None, name:str=None) -> 'PolygonObstacle':
        return cls(PlainPolygon(vertices), motion_model, id_, name)

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.centroid()

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.geometry.angle])
    
    def inflate(self, margin: float, method:str='mitre') -> None:
        """Inflate the obstacle by the given margin."""
        self.inflated_geometry = self.geometry.inflate(margin, method)

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self._motion_model is not None:
            new_state = self._motion_model(self.state, action, dt)
            d_x = new_state[0] - self.state[0]
            d_y = new_state[1] - self.state[1]
            d_angle = new_state[2] - self.state[2]
            new_geo = self.geo_tools.translate(self.geometry, (d_x, d_y))
            new_geo = self.geo_tools.rotate(new_geo, d_angle, origin='center')
            self.geometry = new_geo

    def plot(self, ax: Axes, inflated=False, fill=False, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        if inflated:
            self.inflated_geometry.plot(ax, fill, **kwargs)
        else:
            self.geometry.plot(ax, fill, **kwargs)


class EllipseObstacle(Obstacle):
    def __init__(self, geometry: PlainEllipse, motion_model:Callable=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "ellipse", motion_model, id_, name)
        self.geometry: PlainEllipse

    def __call__(self) -> Tuple[tuple, tuple, float]:
        return self.geometry()
    
    @classmethod
    def from_raw(cls, center:tuple, radii:tuple, angle:float, motion_model:Callable=None, id_:int=None, name:str=None) -> 'EllipseObstacle':
        return cls(PlainEllipse(center, radii, angle), motion_model, id_, name)

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.center()

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.geometry.angle])
    
    def inflate(self, margin: float) -> None:
        """Inflate the obstacle by the given margin."""
        self.inflated_geometry = self.geometry.inflate(margin)

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self._motion_model is not None:
            new_state = self._motion_model(self.state, action, dt)
            new_center = tuple(new_state[:2])
            new_angle = new_state[2]
            self.geometry = PlainEllipse(new_center, self.geometry.radii, new_angle, self.geometry._n_approx)

    def plot(self, ax: Axes, approx:bool=True, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        if approx:
            approx_poly = self.geometry.return_polygon_approximation()
            approx_poly.plot(ax, **kwargs)
        else:
            self.geometry.plot(ax, approx, **kwargs)
        direction_point = (self.geometry.center[0] + self.geometry.radii[0]*np.cos(self.geometry.angle),
                           self.geometry.center[1] + self.geometry.radii[0]*np.sin(self.geometry.angle))
        ax.arrow(self.geometry.center[0], self.geometry.center[1], 
                 direction_point[0] - self.geometry.center[0], direction_point[1] - self.geometry.center[1], 
                 head_width=0.1, head_length=0.1, fc='k', ec='k')


class CircleObstacle(Obstacle):
    def __init__(self, geometry: PlainCircle, motion_model:Callable=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "circle", motion_model, id_, name)
        self.geometry: PlainCircle
        self.angle = None

    def __call__(self) -> Tuple[tuple, float]:
        return self.geometry()
    
    @classmethod
    def from_raw(cls, center:tuple, radius:float, motion_model:Callable=None, id_:int=None, name:str=None) -> 'CircleObstacle':
        return cls(PlainCircle(center, radius), motion_model, id_, name)

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.center

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.angle])

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self._motion_model is not None:
            new_state = self._motion_model(self.state, action, dt)
            new_center = tuple(new_state[:2])
            new_angle = new_state[2]
            self.geometry = PlainCircle(new_center, self.geometry.radius, new_angle, self.geometry._n_approx)

    def plot(self, ax: Axes, approx:bool=True, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        if approx:
            approx_poly = self.geometry.return_polygon_approximation()
            approx_poly.plot(ax, **kwargs)
        else:
            self.geometry.plot(ax, approx, **kwargs)
        direction_point = (self.geometry.center[0] + self.geometry.radius*np.cos(self.geometry.angle),
                           self.geometry.center[1] + self.geometry.radius*np.sin(self.geometry.angle))
        ax.arrow(self.geometry.center[0], self.geometry.center[1], 
                 direction_point[0] - self.geometry.center[0], direction_point[1] - self.geometry.center[1], 
                 head_width=0.1, head_length=0.1, fc='k', ec='k')


