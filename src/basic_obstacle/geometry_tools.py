import math

import shapely.affinity
from shapely.geometry.base import BaseGeometry

from .geometry_plain import PlainGeometry
from .geometry_plain import PlainPoint
from .geometry_plain import PlainPolygon
from .geometry_plain import PlainCircle
from .geometry_plain import PlainEllipse

from typing import Union


class GeometryTools:
    """This class provides some geometry tools for certain objects: points, polygons, circles, ellipses."""

    def translate(self, geometry: PlainGeometry, translation: tuple) -> PlainGeometry:
        """Translate a geometry by a given translation vector."""
        if isinstance(geometry, PlainPoint):
            return self.point_translate(geometry, translation)
        elif isinstance(geometry, PlainPolygon):
            return self.polygon_translate(geometry, translation)
        elif isinstance(geometry, PlainCircle):
            return self.circle_translate(geometry, translation)
        elif isinstance(geometry, PlainEllipse):
            return self.ellipse_translate(geometry, translation)
        else:
            raise ValueError(f"Unknown geometry type: {type(geometry)}")
        
    def rotate(self, geometry: PlainGeometry, angle: float, origin:Union[tuple, str]='center') -> PlainGeometry:
        """Rotate a geometry by a given angle around a given origin. The `angle` is in radians."""
        if isinstance(geometry, PlainPoint):
            return self.point_rotate(geometry, angle, origin)
        elif isinstance(geometry, PlainPolygon):
            return self.polygon_rotate(geometry, angle, origin)
        elif isinstance(geometry, PlainCircle):
            return self.circle_rotate(geometry, angle, origin)
        elif isinstance(geometry, PlainEllipse):
            return self.ellipse_rotate(geometry, angle, origin)
        else:
            raise ValueError(f"Unknown geometry type: {type(geometry)}")

    def frame_transform(self, geometry: PlainGeometry, origin_A: tuple, origin_B: tuple, angle_A: float, angle_B: float) -> PlainGeometry:
        """Transform a geometry from a given frame A to a given frame B.
        Frames A and B are defined by the relative origins and the angles to the standard world frame.

        This is equivalent to:
            First translate along `vector(BA)` and rotate `-angle(B-A)` to frame A.
        """
        if isinstance(geometry, (PlainPoint, PlainPolygon)):
            return self.shapely_frame_transform(geometry.to_shapely(), origin_A, origin_B, angle_A, angle_B)
        elif isinstance(geometry, PlainCircle):
            raise NotImplementedError
        elif isinstance(geometry, PlainEllipse):
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown geometry type: {type(geometry)}")

    ### Translate ###
    def point_translate(self, point: PlainPoint, translation: tuple) -> PlainPoint:
        """Translate a point by a given translation vector."""
        return PlainPoint(point.x + translation[0], point.y + translation[1])
    
    def polygon_translate(self, polygon: PlainPolygon, translation: tuple) -> PlainPolygon:
        """Translate a polygon by a given translation vector."""
        return PlainPolygon([self.point_translate(vertex, translation) for vertex in polygon.vertices])
    
    def circle_translate(self, circle: PlainCircle, translation: tuple) -> PlainCircle:
        """Translate a circle by a given translation vector."""
        return PlainCircle(self.point_translate(circle.center, translation), circle.radius)
    
    def ellipse_translate(self, ellipse: PlainEllipse, translation: tuple) -> PlainEllipse:
        """Translate an ellipse by a given translation vector."""
        return PlainEllipse(self.point_translate(ellipse.center, translation), ellipse.radii, ellipse.angle)
    
    ### Rotate [shapely] ###
    def point_rotate(self, point: PlainPoint, angle: float, origin:Union[tuple, str]='center') -> PlainPoint:
        """Rotate a point by a given angle around a given origin."""
        return PlainPoint.from_shapely(self.shapely_rotate(point.to_shapely(), angle, origin))

    def polygon_rotate(self, polygon: PlainPolygon, angle: float, origin:Union[tuple, str]='center') -> PlainPolygon:
        """Rotate a polygon by a given angle around a given origin."""
        return PlainPolygon.from_shapely(self.shapely_rotate(polygon.to_shapely(), angle, origin))
    
    def circle_rotate(self, circle: PlainCircle, angle: float, origin:Union[tuple, str]='center') -> PlainCircle:
        """Rotate a circle by a given angle around a given origin."""
        return PlainCircle(self.point_rotate(circle.center, angle, origin), circle.radius)
    
    def ellipse_rotate(self, ellipse: PlainEllipse, angle: float, origin:Union[tuple, str]='center') -> PlainEllipse:
        """Rotate an ellipse by a given angle around a given origin."""
        return PlainEllipse(self.point_rotate(ellipse.center, angle, origin), ellipse.radii, ellipse.angle + angle)
        
    ### Shapely backend ###
    @staticmethod
    def shapely_translate(geometry: BaseGeometry, translation: tuple) -> BaseGeometry:
        """Translate a shapely geometry by a given translation vector."""
        return shapely.affinity.translate(geometry, translation[0], translation[1])

    @staticmethod
    def shapely_rotate(geometry: BaseGeometry, angle: float, origin:Union[tuple, str]='center') -> BaseGeometry:
        """Rotate a shapely geometry by a given angle around a given origin."""
        return shapely.affinity.rotate(geometry, angle, origin=origin, use_radians=True)

    @staticmethod
    def shapely_frame_transform(geometry: BaseGeometry, origin_A: tuple, origin_B: tuple, angle_A: float, angle_B: float) -> BaseGeometry:
        """Transform a shapely geometry from a given frame A to a given frame B.
        Frames A and B are defined by the relative origins and the angles to the standard world frame.

        This is equivalent to:
            First translate along `vector(BA)` and rotate `-angle(B-A)` to frame A.
        """
        rotation_angle = angle_A - angle_B
        translation_vector = (origin_A[0] - origin_B[0], origin_A[1] - origin_B[1])
        return GeometryTools.shapely_rotate(GeometryTools.shapely_translate(geometry, translation_vector), rotation_angle)

    @staticmethod
    def shapely_affine_transform(geometry: BaseGeometry, angle: float, translation: tuple) -> BaseGeometry:
        """Rotate a shapely geometry by a given angle around a given origin.

        This is equivalent to (1) or (2):

            (1): translate a rotated geometry by a given translation vector.

            (2): rotate a translated geometry by a given angle around the translation vector point.
        """
        transform_mtx_param = [math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle), translation[0], translation[1]]
        return shapely.affinity.affine_transform(geometry, transform_mtx_param)


