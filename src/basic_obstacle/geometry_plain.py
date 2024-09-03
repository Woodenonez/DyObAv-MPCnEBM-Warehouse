from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import math
import matplotlib.patches as patches # type: ignore

from shapely.geometry import Point, Polygon # type: ignore
from shapely.geometry import JOIN_STYLE # type: ignore

from matplotlib.axes import Axes # type: ignore


PathNode = tuple[float, float]


class PlainGeometry(ABC):
    """A plain geometry class without any dependencies."""
    @abstractmethod
    def __call__(self):
        """The call method returns the geometry in a native python (e.g. tuple) format."""
        pass
    
    def inflate(self, margin: float, *args):
        """Inflate the geometry by a given margin."""
        raise NotImplementedError
    
    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the geometry on the given axes."""
        raise NotImplementedError


@dataclass
class PlainPoint(PlainGeometry):
    """A plain point class without any dependencies."""
    x: float
    y: float

    def __call__(self) -> PathNode:
        return (self.x, self.y)

    def __getitem__(self, idx) -> float:
        return (self.x, self.y)[idx]

    def __sub__(self, other_point:'PlainPoint') -> float:
        return math.hypot(self.x-other_point.x, self.y-other_point.y)
    
    @classmethod
    def from_shapely(cls, point: Point) -> 'PlainPoint':
        """Create a PlainPoint from a shapely Point."""
        return cls(point.x, point.y)

    def to_shapely(self) -> Point:
        """Create a shapely Point from a PlainPoint."""
        return Point(self.x, self.y)
    
    def inflate(self, margin: float, n:int=4) -> 'PlainPoint': # type: ignore
        """[shapely] Inflate a point (to a regular polygon) by a given positive margin.
        
        The number of vertices of the polygon is given by 4*n.
        """
        inflated_shapely = self.to_shapely().buffer(margin, resolution=n)
        return PlainPoint.from_shapely(inflated_shapely)

    def plot(self, ax: Axes, **kwargs) -> None:
        ax.plot(self.x, self.y, **kwargs)

@dataclass
class PlainPolygon(PlainGeometry):
    """A plain polygon class without any dependencies.
    Certain functions need the shapely library."""
    vertices: list[PlainPoint]
    angle: float = 0

    @property
    def ngon(self) -> int: # the number of vertices
        return len(self.vertices)
    
    @property
    def centroid(self) -> PlainPoint:
        """[shapely] Return the centroid of the polygon."""
        return PlainPoint.from_shapely(self.to_shapely().centroid)

    def __call__(self) -> list[PathNode]:
        return [x() for x in self.vertices]

    def __getitem__(self, idx) -> PlainPoint:
        return self.vertices[idx]

    @classmethod
    def from_list_of_tuples(cls, vertices: list[PathNode]):
        return cls([PlainPoint(*v) for v in vertices]) 
    
    @classmethod
    def from_shapely(cls, polygon: Polygon, angle:float=0) -> 'PlainPolygon':
        """Create a PlainPolygon from a shapely Polygon."""
        return cls([PlainPoint(*v) for v in polygon.exterior.coords[:-1]], angle=angle)

    def to_shapely(self) -> Polygon:
        """Create a shapely Polygon from a PlainPolygon."""
        return Polygon(self())
    
    def inflate(self, margin:float,  method:Union[JOIN_STYLE, str]=JOIN_STYLE.mitre) -> 'PlainPolygon': # type: ignore
        """[shapely] Return a new inflated polygon.
        If the margin is negative, the polygon will be deflated."""
        if isinstance(method, str):
            if method == 'round':
                method = JOIN_STYLE.round
            elif method == 'mitre':
                method = JOIN_STYLE.mitre
            elif method == 'bevel':
                method = JOIN_STYLE.bevel
            else:
                raise ValueError(f"The given method must be one of 'round', 'mitre', or 'bevel'.")
        inflated_shapely = self.to_shapely().buffer(margin,join_style=method)
        return PlainPolygon.from_shapely(inflated_shapely, angle=self.angle)
    
    def distance_to_point(self, point: PlainPoint) -> float:
        """[shapely] Return the distance between the polygon and the given point. 
        Positive if the point is outside the polygon, negative if inside."""
        distance = self.to_shapely().distance(point.to_shapely())
        if distance > 0:
            return distance
        else:
            exteriors = self.to_shapely().exterior
            return -exteriors.distance(point)
        
    def contains_point(self, point: PlainPoint) -> bool:
        """Check if the polygon contains the point."""
        return self.to_shapely().contains(point.to_shapely())

    def plot(self, ax: Axes, fill=False, **kwargs) -> None:
        """Plot the polygon on the given axes."""
        plot_vertices = self() + [self()[0]]
        if fill:
            ax.fill(*zip(*plot_vertices), **kwargs)
        else:
            ax.plot(*zip(*plot_vertices), **kwargs)

@dataclass
class PlainEllipse(PlainGeometry):
    """A plain ellipse class without any dependencies.

    Methods:
        return_polygon_approximation: return a (inscribed) polygon approximation of the ellipse
        contains_point: check if the ellipse contains the point
    """
    center: PlainPoint
    radii: tuple[float, float]
    angle: float

    def __call__(self) -> tuple[tuple, tuple, float]:
        return (self.center(), self.radii, self.angle)
    
    def inflate(self, margin:float) -> 'PlainEllipse': # type: ignore
        return PlainEllipse(self.center, (self.radii[0]+margin, self.radii[1]+margin), self.angle)
    
    def distance_to_point(self, point: PlainPoint) -> float:
        """Return the distance (approximated) between the ellipse and the given point. 
        Positive if the point is outside the ellipse, negative if inside."""
        approximated_poly = self.return_polygon_approximation(100)
        return approximated_poly.distance_to_point(point)
        
    def contains_point(self, point: PlainPoint, value:bool=False) -> Union[bool, float]:
        """Check if the ellipse contains the point. 
        
        If `value` is True, return a value. Positive value means the point is inside the ellipse."""
        x, y = self.center()
        rx, ry = self.radii
        a = self.angle
        rotation_matrix = [[math.cos(a), -math.sin(a)], 
                           [math.sin(a), math.cos(a)]]
        pt = PlainPoint(rotation_matrix[0][0]*(point.x-x) + rotation_matrix[0][1]*(point.y-y), 
                        rotation_matrix[1][0]*(point.x-x) + rotation_matrix[1][1]*(point.y-y))
        if value:
            return 1 - (pt.x/rx)**2 - (pt.y/ry)**2
        return (pt.x/rx)**2 + (pt.y/ry)**2 <= 1

    def return_polygon_approximation(self, n:int=10) -> PlainPolygon:
        """Return a polygon approximation of the ellipse."""
        x, y = self.center()
        rx, ry = self.radii
        a = self.angle
        ellipse_samples_raw = [(rx*math.cos(2*math.pi*i/n), ry*math.sin(2*math.pi*i/n)) for i in range(n)]
        rotation_matrix = [[math.cos(a), -math.sin(a)], 
                           [math.sin(a), math.cos(a)]]
        ellipse_samples = [PlainPoint(x + rotation_matrix[0][0]*sample[0] + rotation_matrix[0][1]*sample[1], 
                                      y + rotation_matrix[1][0]*sample[0] + rotation_matrix[1][1]*sample[1]) for sample in ellipse_samples_raw]
        return PlainPolygon(ellipse_samples)
    
    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the ellipse on the given axes."""
        ax.add_patch(patches.Ellipse(self.center(), self.radii[0]*2, self.radii[1]*2, self.angle, **kwargs))


@dataclass
class PlainCircle(PlainGeometry):
    """A plain circle class without any dependencies.

    Methods:
        `return_polygon_approximation`: return a (inscribed-default/circumscribed) polygon approximation of the circle
        `contains_point`: check if the circle contains the point
    """
    center: PlainPoint
    radius: float

    def __call__(self) -> tuple[tuple, float]:
        return (self.center(), self.radius)

    def inflate(self, margin:float) -> 'PlainCircle': # type: ignore
        return PlainCircle(self.center, self.radius+margin)
    
    def distance_to_point(self, point: PlainPoint) -> float:
        """Return the distance between the circle and the given point. 
        Positive if the point is outside the circle, negative if inside."""
        distance_to_center = self.center - point
        return distance_to_center - self.radius
        
    def contains_point(self, point: PlainPoint, value:bool=False) -> Union[bool, float]:
        """Check if the circle contains the point. 

        If `value` is True, 
        return the difference of (i) the distance from the point to the certer and (ii) the radius.
        """
        if value:
            return math.hypot(self.center.x-point.x, self.center.y-point.y) - self.radius
        return math.hypot(self.center.x-point.x, self.center.y-point.y) <= self.radius

    def return_polygon_approximation(self, n:int=10, inscribed:bool=True) -> PlainPolygon:
        """Return a polygon approximation of the circle. If not inscribed, it is circumscribed."""
        if inscribed:
            vertices = [PlainPoint(self.center.x + self.radius*math.cos(2*math.pi/n*i), 
                                   self.center.y + self.radius*math.sin(2*math.pi/n*i)) for i in range(n)]
        else:
            vertices = [PlainPoint(self.center.x + (self.radius/math.cos(math.pi/n))*math.cos(2*math.pi/n*i), 
                                   self.center.y + (self.radius/math.cos(math.pi/n))*math.sin(2*math.pi/n*i)) for i in range(n)]
        return PlainPolygon(vertices)
    
    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the circle on the given axes."""
        ax.add_patch(patches.Circle(self.center(), self.radius, **kwargs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt # type: ignore

    list_of_points_raw = [(1,2), (1,1), (1,0), (0,1)]

    list_of_points = [PlainPoint(*v) for v in list_of_points_raw]
    polygon = PlainPolygon(list_of_points)
    polygon_inflated = polygon.inflate(0.1, 'mitre')

    circle = PlainCircle(PlainPoint(0,0), 1)
    circle_approx = circle.return_polygon_approximation(100)
    polygon_inscribed = circle.return_polygon_approximation(6, inscribed=True)
    polygon_circumscribed = circle.return_polygon_approximation(6, inscribed=False)

    ellipse = PlainEllipse(PlainPoint(-1,-2), (1,2), math.pi/4)
    ellipse_approx = ellipse.return_polygon_approximation(100)

    print(ellipse)

    _, ax = plt.subplots()
    polygon.plot(ax, fill=False, marker='o', linestyle='-', color='b')
    polygon_inflated.plot(ax, fill=False, marker='o', linestyle='-', color='r')

    plt.plot(*zip(*circle_approx()), '-')
    plt.plot(*zip(*polygon_inscribed()), 'x-')
    plt.plot(*zip(*polygon_circumscribed()), 'o-')

    plt.plot(*zip(*ellipse_approx()), '-')

    plt.axis('equal')
    plt.show()