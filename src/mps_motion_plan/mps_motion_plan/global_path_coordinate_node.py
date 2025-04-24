import os
from typing import Optional, Union

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory # type: ignore

from geometry_msgs.msg import Point, Polygon, Point32 # type: ignore
from mps_interfaces.srv import GetRobotSchedule # type: ignore
from mps_interfaces.srv import GetInflatedMap # type: ignore
# from mps_interfaces.msg import Schedule, ScheduleArray, ScheduleArrayStamped # type: ignore
from map_interfaces.msg import GeometricMap as MapMsg # type: ignore
from map_interfaces.msg import ObjectShapeType, ObjectMotionType, ObjectSemanticType # type: ignore
from map_interfaces.msg import PolygonObject, PolygonObjectArray # type: ignore

from visualization_msgs.msg import Marker, MarkerArray # type: ignore

from .global_path_coordinate import GlobalPathCoordinator
from .basic_components.map_geometric import GeometricMap
from .basic_components.geometry_plain import PlainPolygon
from .map_tf import ScaleOffsetReverseTransform


# INFLATION_MARGIN = 0.2 # warehouse
INFLATION_MARGIN = 0.3 # zospital



class GlobalPathCoordinateNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        pkg_root_dir = get_package_share_directory('mps_motion_plan')

        self.declare_parameter('rescale_map', False)
        self.rescale_map = self.get_parameter('rescale_map').value

        self.declare_parameter('rescale_graph', False)
        self.rescale_graph = self.get_parameter('rescale_graph').value

        self.declare_parameter('graph_file_name', 'graph.json')
        self.graph_file_name = self.get_parameter('graph_file_name').value
        self.json_path = os.path.join(pkg_root_dir, 'data', self.graph_file_name)

        self.declare_parameter('schedule_file_name', 'schedule.csv')
        self.schedule_file_name = self.get_parameter('schedule_file_name').value
        self.csv_path = os.path.join(pkg_root_dir, 'data', self.schedule_file_name)

        self.tf_img2real = ScaleOffsetReverseTransform(
            scale=0.1, 
            offsetx_after=-15.0, 
            offsety_after=-15.0, 
            y_reverse=(not False), 
            y_max_before=293) # global_setting_warehouse.yaml

        self.coordinator = None
        if not os.path.exists(self.json_path):
            self.get_logger().warn(f"Graph file {self.json_path} does not exist.")
        elif not os.path.exists(self.csv_path):
            self.get_logger().warn(f"Schedule file {self.csv_path} does not exist.")
        else:
            self._init_coordinator()

        self.timer = self.create_timer(0.5, self.timer_callback)
        
        map_info_name = 'geometry_map'
        self.map_info_subscriber = self.create_subscription(
            MapMsg,
            map_info_name,
            self.map_info_callback, # event driven
            10
        )

        inflated_map_vis_name = 'inflated_map_vis'
        self.inflated_map_vis_publisher = self.create_publisher(
            MarkerArray, 
            inflated_map_vis_name, 
            10
        )

        self._init_map_info()
        map_service_name = 'get_inflated_map'
        self.map_service = self.create_service(
            GetInflatedMap, 
            map_service_name, 
            self.get_inflated_map_callback
        )

        service_name = 'get_robot_schedule'
        self.srv = self.create_service(
            GetRobotSchedule, 
            service_name, 
            self.get_robot_schedule_callback
        )

        self.boundary_coords = None
        self.obstacle_list = None
        self.map_updated = False
        self.map_received = False

        self.color_node = (0.0, 1.0, 0.0) # green
        self.color_edge = (1.0, 1.0, 1.0) # white

    def _init_coordinator(self):
        self.coordinator = GlobalPathCoordinator.from_csv(self.csv_path)
        self.coordinator.load_graph_from_json(self.json_path)
        if self.rescale_graph:
            self.coordinator.current_graph.graph_coords_cvt(self.tf_img2real)

    def _init_map_info(self):
        self.inflated_map_msg = MapMsg()
        self.inflated_map_msg.header.frame_id = 'map'
        self.inflated_map_msg.header.stamp = self.get_clock().now().to_msg()

    def timer_callback(self):
        if self.map_received:
            map_vis_msg = self.geo_map_to_vis_msg(self.inflated_map_msg, 
                                                #   boundary_color=self.boundary_color, 
                                                obstacle_color=(1.0, 0.1, 0.1))
            assert isinstance(map_vis_msg, MarkerArray), f"The map_vis_msg is not a MarkerArray, but {type(map_vis_msg)}."
            self.inflated_map_vis_publisher.publish(map_vis_msg)
            self.get_logger().debug("Inflated map visualization published.")

    def map_info_callback(self, msg: MapMsg):
        """Callback function for the map info subscriber.

        Args:
            msg: The map info message, GeometricMap type.

        Attributes:
            boundary_coords: List of tuples of boundary coordinates.
            obstacle_list: List of list of tuples of obstacle coordinates.
        """
        self.get_logger().debug("Map info received.")

        ### Map info message: GeometricMap ###
        boundary_msg = msg.boundary
        obstacle_msg = msg.obstacle_list

        new_boundary_coords = self.polygon_to_tuples(boundary_msg.polygon)
        if self.boundary_coords is None or new_boundary_coords != self.boundary_coords:
            self.boundary_coords = new_boundary_coords
            self.map_updated = True
            self.get_logger().debug("Boundary updated.")

        new_obstacle_list = []
        obstacle_objects:list[PolygonObject] = obstacle_msg.polygon_objects
        for obstacle_obj in obstacle_objects:
            new_obstacle_list.append(self.polygon_to_tuples(obstacle_obj.polygon))
        if self.obstacle_list is None or new_obstacle_list != self.obstacle_list:
            self.obstacle_list = new_obstacle_list
            self.map_updated = True
            self.get_logger().debug("Obstacle list updated.")

        if self.map_updated and self.coordinator is not None:
            self.coordinator.load_map(self.boundary_coords, 
                                      self.obstacle_list, 
                                      inflation_margin=INFLATION_MARGIN)
            if self.rescale_map:
                self.coordinator.current_map.map_coords_cvt(self.tf_img2real)
                self.coordinator.inflated_map.map_coords_cvt(self.tf_img2real)
            
            boundary_msg, obstacle_msg = self.geo_map_to_msg(self.coordinator.inflated_map)
            self.inflated_map_msg.boundary = boundary_msg
            self.inflated_map_msg.obstacle_list = obstacle_msg
            self.map_updated = False

        self.map_received = True

    def get_inflated_map_callback(self, request: GetInflatedMap.Request, response: GetInflatedMap.Response):
        """Service to get the inflated map."""
        if self.coordinator is None:
            self.get_logger().warn(f"The coordinator is not initialized.")
            return response
        if not self.map_received:
            self.get_logger().warn(f"The map is not received yet.")
            return response
        robot_id = request.robot_id
        self.get_logger().info(f"Request (for the inflated map) ID: {robot_id}. Success.")

        self.inflated_map_msg.header.stamp = self.get_clock().now().to_msg()
        response.inflated_map = self.inflated_map_msg
        return response

    def get_robot_schedule_callback(self, request: GetRobotSchedule.Request, response: GetRobotSchedule.Response):
        """Service to get the robot schedule.

        Notes:
            The coordinator must be initialized before calling this service.
            To test this service: ros2 service call /get_robot_schedule mps_interfaces/srv/GetRobotSchedule "{robot_id: 0, current_time: 0.0}"
        """
        if self.coordinator is None:
            self.get_logger().warn(f"The coordinator is not initialized.")
            return response
        robot_id = request.robot_id
        current_time = request.current_time
        try:
            path_coords, path_times = self.coordinator.get_robot_schedule(robot_id, time_offset=current_time)
            self.get_logger().info(f"Request (for the schedule) ID: {robot_id}. Success.")
        except KeyError:
            self.get_logger().info(f"Request (for the schedule) ID: {robot_id}. Failed. Check available IDs: {self.coordinator.robot_ids}.")
            return response

        self.get_logger().info(f"Robot {robot_id} schedule: {path_coords}, {path_times}.")
        
        response.path_schedule.header.stamp = self.get_clock().now().to_msg()
        response.path_schedule.header.frame_id = "map"
        response.path_schedule.path_schedule.robot_id = robot_id
        response.path_schedule.path_schedule.path_coords = [Point(x=x_[0], y=x_[1]) for x_ in path_coords]
        if path_times is not None:
            response.path_schedule.path_schedule.path_times = path_times
        else:
            response.path_schedule.path_schedule.path_times = [-1.0]
        return response
    
    def geo_map_to_msg(self, geo_map: GeometricMap) -> tuple[PolygonObject, PolygonObjectArray]:
        """Convert a GeometricMap object to a PolygonObjectArray message"""
        boundary_msg = PolygonObject()
        obstacle_msg = PolygonObjectArray()

        motion_type_msg = ObjectMotionType()
        motion_type_msg.o_motion_type = ObjectMotionType.STATIC
        shape_type_msg = ObjectShapeType()
        shape_type_msg.o_shape_type = ObjectShapeType.POLYGON
        semantic_type_msg = ObjectSemanticType()
        semantic_type_msg.o_semantic_type = ObjectSemanticType.UNKNOWN

        boundary_msg.object_id = -1
        boundary_msg.object_name = "boundary"
        boundary_msg.polygon = self.tuples_to_polygon_msg(geo_map.boundary_coords)

        for obs_id in geo_map.id_list:
            obs_info = geo_map.get_obstacle_info(obs_id)
            obs_msg = PolygonObject()
            obs_msg.object_id = obs_info['id_']
            obs_msg.object_name = obs_info['name']
            obs_msg.polygon = self.tuples_to_polygon_msg(obs_info['vertices'])
            
            plain_poly = PlainPolygon.from_list_of_tuples(obs_info['vertices'])
            plain_poly_centroid = plain_poly.centroid()
            obs_msg.centroid = Point32(x=float(plain_poly_centroid[0]), y=float(plain_poly_centroid[1]), z=0.0)
            obs_msg.object_motion_t = motion_type_msg
            obs_msg.object_shape_t = shape_type_msg
            obs_msg.object_semantic_t = semantic_type_msg

            obstacle_msg.polygon_objects.append(obs_msg)
        return boundary_msg, obstacle_msg
    
    def geo_map_to_vis_msg(self, map_msg: MapMsg, 
                           name_space:str="inflated_map_ns", 
                           id_start:int=0, 
                        #    boundary_color=(0.0, 0.0, 0.0),
                           obstacle_color=(1.0, 1.0, 1.0),
                           scale:float=0.05) -> MarkerArray:
        """Convert a PolygonObjectArrayStamped message to a MarkerArray message"""
        marker_all_poly_msg = MarkerArray()
        marker_id = id_start
        # boundary_msg = map_msg.boundary
        obstacle_msg = map_msg.obstacle_list

        # poly_points = boundary_msg.polygon.points
        # poly_points.append(poly_points[0])
        # marker_poly_msg = self.points_to_vis_msg(poly_points, boundary_color, marker_id, scale)
        # marker_poly_msg.header = map_msg.header
        # marker_poly_msg.ns = name_space
        # marker_all_poly_msg.markers.append(marker_poly_msg)

        polygon_objects:list[PolygonObject] = obstacle_msg.polygon_objects
        for poly_obj in polygon_objects:
            assert isinstance(poly_obj, PolygonObject), f"The object is not a PolygonObject, but {type(poly_obj)}."
            marker_id += 1
            poly_points = poly_obj.polygon.points
            poly_points.append(poly_points[0])
            marker_poly_msg = self.points_to_vis_msg(poly_points, obstacle_color, marker_id, scale)
            marker_poly_msg.header = map_msg.header
            marker_poly_msg.ns = name_space
            marker_all_poly_msg.markers.append(marker_poly_msg)
        return marker_all_poly_msg

    @staticmethod
    def polygon_to_tuples(polygon: Polygon) -> list[tuple[float, float]]:
        """Convert a Polygon message to a list of tuples"""
        vertices = []
        points:list[Point32] = polygon.points
        for point in points:
            vertices.append((point.x, point.y))
        return vertices
    
    @staticmethod
    def tuples_to_polygon_msg(vertices: list[tuple[float, float]]) -> Polygon:
        """Convert a list of vertices to a Polygon message"""
        polygon_msg = Polygon()
        for vertex in vertices:
            point_msg = Point32(x=float(vertex[0]), y=float(vertex[1]), z=0.0)
            polygon_msg.points.append(point_msg)
        return polygon_msg
    
    @staticmethod
    def points_to_vis_msg(points: list[Point], 
                          color: tuple[float, float, float],
                          marker_id: int,
                          scale:float=0.05) -> Marker:
        """Convert a pair of points message to a Marker message"""
        marker_msg = Marker()
        marker_msg.id = marker_id
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = 0.0
        marker_msg.pose.position.y = 0.0
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.scale.x = scale
        marker_msg.scale.y = scale
        marker_msg.scale.z = scale

        marker_msg.color.a = 0.5
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]

        marker_msg.lifetime.sec = 0 # forever
        marker_msg.frame_locked = False # not locked to a frame
        [marker_msg.points.append(Point(x=point.x, y=point.y, z=point.z)) for point in points]
        return marker_msg


def main(args=None):
    rclpy.init(args=args)

    motion_plan_node = GlobalPathCoordinateNode()
    rclpy.spin(motion_plan_node)
    motion_plan_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()