from typing import Any, Optional, Callable

import pandas as pd # type: ignore
import networkx as nx # type: ignore
from matplotlib.axes import Axes # type: ignore

from .path_plan_graph import dijkstra

from basic_map.graph import NetGraph
from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap
from basic_obstacle.geometry_plain import PlainPolygon


PathNode = tuple[float, float]


class GlobalPathCoordinator:
    """Recieve the schedule of all robots and return the path and time of a specific robot.

    Attributes:
        total_schedule: The total schedule for all robots.
        robot_ids: The ids of all robots.
    
    Notes:
        Load the graph before calling `get_robot_schedule`.
    """
    def __init__(self, total_schedule: pd.DataFrame) -> None:
        """
        Args:
            total_schedule: The total schedule for all robots.

        Notes:
            A total schedule is a dataframe with the following columns:
            - `robot_id`: The id of the robot.
            - `node_id`: The path node id of the robot.
            - `ETA`: The estimated time of arrival at the node.

            Or:
            - `robot_id`: The id of the robot.
            - `start`: The start node id of the robot.
            - `end`: The end node id of the robot.
            - `EDT`(optional): The estimated duration of travel, if not provided, the time plan will be None.
        """
        self._total_schedule = total_schedule
        self._robot_ids = total_schedule['robot_id'].unique().tolist()
        self.robot_schedule_dict = {}

        for robot_id in self._robot_ids:
            robot_schedule:pd.DataFrame = self.total_schedule[self.total_schedule['robot_id'] == robot_id]
            robot_schedule = robot_schedule.reset_index(drop=True)
            self.robot_schedule_dict[robot_id] = robot_schedule

        self._G:Optional[NetGraph] = None
        self.img_map: Optional[OccupancyMap] = None

    @property
    def total_schedule(self) -> pd.DataFrame:
        return self._total_schedule
    
    @property
    def robot_ids(self) -> list:
        return self._robot_ids
    
    @property
    def current_map(self):
        return self._current_map
    
    @property
    def inflated_map(self):
        return self._inflated_map
    
    @property
    def current_graph(self):
        return self._G
    
    @classmethod
    def from_csv(cls, csv_path: str, csv_sep:str=','):
        """Load the total schedule from a csv file."""
        total_schedule = pd.read_csv(csv_path, sep=csv_sep, header=0)
        return cls(total_schedule)
    
    @classmethod
    def from_dict(cls, schedule_dict: dict):
        """Load the total schedule from a dictionary."""
        total_schedule = pd.DataFrame(schedule_dict)
        return cls(total_schedule)

    @staticmethod
    def inflate_map(original_map: GeometricMap, inflation_margin: float):
        boundary_coords, obstacle_coords_list = original_map()
        for i, obs in enumerate(obstacle_coords_list):
            inflated_obs = PlainPolygon.from_list_of_tuples(obs).inflate(inflation_margin)
            obstacle_coords_list[i] = inflated_obs()
        boundary_polygon = PlainPolygon.from_list_of_tuples(boundary_coords).inflate(-inflation_margin)
        boundary_coords = boundary_polygon()
        return GeometricMap.from_raw(boundary_coords, obstacle_coords_list)

    def load_graph(self, G: NetGraph):
        self._G = G

    def load_graph_from_json(self, json_path: str):
        self.load_graph(NetGraph.from_json(json_path))

    def load_map(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]], rescale:Optional[float]=None, inflation_margin:Optional[float]=None):
        self._current_map = GeometricMap.from_raw(boundary_coords, obstacle_list, rescale=rescale)
        if inflation_margin is not None:
            self._inflated_map = self.inflate_map(self._current_map, inflation_margin)
        else:
            self._inflated_map = self._current_map

    def load_map_from_json(self, json_path: str, rescale:Optional[float]=None, inflation_margin:Optional[float]=None):
        self._current_map = GeometricMap.from_json(json_path, rescale=rescale)
        boundary_coords, obstacle_coords_list = self._current_map()
        self.load_map(boundary_coords, obstacle_coords_list, rescale=None, inflation_margin=inflation_margin)

    def load_img_map(self, img_path: str):
        self.img_map = OccupancyMap.from_image(img_path)

    def coordinate_convert(self, ct: Callable):
        """Convert the coordinates of the map and the graph.

        Args:
            ct: The coordinate conversion function.
        """
        self.current_map.map_coords_cvt(ct)
        self.inflated_map.map_coords_cvt(ct)
        self.current_graph.graph_coords_cvt(ct)


    def get_schedule_with_node_index(self, robot_id: int) -> tuple[list, Optional[list[float]], bool]:
        """Get the schedule of a robot.
        
        Returns:
            path_nodes: The path nodes of the robot.
            path_times: The path times of the robot, None if not provided.
            whole_path: Whether the path is complete.

        Notes:
            This method is called within `get_robot_schedule`.
        """
        schedule:pd.DataFrame = self.robot_schedule_dict[robot_id]
        if 'robot_id' not in schedule.columns:
            raise ValueError("The schedule must include robot_id.")

        if 'ETA' in schedule.columns:
            path_nodes = schedule['node_id'].tolist()
            path_times = schedule['ETA'].tolist()
            path_times_0 = path_times[0]
            if isinstance(path_times_0, str):
                if path_times_0.lower() == 'none':
                    path_times = None
            whole_path = True
        elif 'EDT' in schedule.columns:
            path_nodes = [schedule['start_node'].iloc[0], schedule['end_node'].iloc[0]]
            path_times = [0.0, schedule['EDT'].iloc[0]]
            whole_path = False
        else:
            path_nodes = [schedule['start_node'].iloc[0], schedule['end_node'].iloc[0]]
            path_times = None
            whole_path = False
        return path_nodes, path_times, whole_path
        
    def get_robot_schedule(self, robot_id: int, time_offset:float=0.0, position_key="position") -> tuple[list[tuple[float, float]], Optional[list[float]]]:
        """
        Args:
            time_offset: The delayed time offset of the schedule.

        Raises:
            ValueError: If the graph is not loaded.
            
        Returns:
            path_coords: list of coordinates of the path nodes
            path_times: list of absolute time stamps, None if not provided.
        """
        if self._G is None:
            raise ValueError("The graph is not loaded.")
        
        path_nodes, path_times, whole_path = self.get_schedule_with_node_index(robot_id)
        
        if whole_path:
            path_coords:list[tuple[float, float]] = [self._G.nodes[node_id][position_key] for node_id in path_nodes]
        if not whole_path:
            source = path_nodes[0]
            target = path_nodes[1]
            path_coords_with_index, section_length_list = self.get_shortest_path(self._G, source, target)
            path_coords = [(x[0], x[1]) for x in path_coords_with_index]
            if path_times is not None:
                edt = path_times[1]
                path_times = [x/sum(section_length_list)*edt for x in section_length_list]
        if path_times is not None:
            path_times = [time_offset + x for x in path_times]
        return path_coords, path_times
    

    def plot_map(self, ax, inflated:bool=False, original_plot_args:dict={'c':'k'}, inflated_plot_args:dict={'c':'r'}):
        self.current_map.plot(ax, inflated, original_plot_args, inflated_plot_args)

    def plot_graph(self, ax, node_style='x', node_text:bool=True, edge_color='r'):
        self.current_graph.plot_graph(ax, node_style, node_text, edge_color)


    @staticmethod
    def get_shortest_path(graph: nx.Graph, source: Any, target: Any, algorithm:str='dijkstra'):
        """
        Args:
            source: The source node ID.
            target: The target node ID.
            algorithm: The algorithm used to find the shortest path. Currently only "dijkstra".

        Returns:
            shortest_path: The shortest path from source to target, each element is (x, y, node_id)
            section_lengths: The lengths of all sections in the shortest path.

        Notes:
            The weight key should be set to "weight" in the graph.
        """
        if algorithm == 'dijkstra':
            planner = dijkstra.DijkstraPathPlanner(graph)
            _, paths = planner.k_shortest_paths_with_coords(source, target, k=1)
            shortest_path = paths[0]
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
        section_lengths:list[float] = [graph.edges[shortest_path[i], shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1)]
        return shortest_path, section_lengths
    

