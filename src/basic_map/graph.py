import json
import math
import random
from typing import Any, Callable

import networkx as nx # type: ignore
from matplotlib.axes import Axes # type: ignore


class NetGraph(nx.Graph):
    """Interactive interface with networkx library.
    
    The function from_json() should be used to load the graph.
    """
    def __init__(self, node_dict: dict[Any, tuple], edge_list: list[tuple]):
        """The init should not be used directly. Use from_json() instead.

        Args:
            node_dict: {node_id: (x, y)}, node_id can be number or string
            edge_list: [(node_id1, node_id2), ...]
        """
        super().__init__()
        self._position_key = 'position'
        for node_id in node_dict:
            self.add_node(int(node_id), **{self._position_key: node_dict[node_id]})
        self.add_edges_from(edge_list)
        self._distance_weight()

    def _distance_weight(self):
        def euclidean_distance(graph: nx.Graph, source, target):
            x1, y1 = graph.nodes[source][self._position_key]
            x2, y2 = graph.nodes[target][self._position_key]
            return math.sqrt((x1-x2)**2 + (y1-y2)**2) 
        for e in self.edges():
            self[e[0]][e[1]]['weight'] = euclidean_distance(self, e[0], e[1])

    @classmethod
    def from_json(cls, json_path:str):
        with open(json_path, 'r') as jf:
            data = json.load(jf)
        node_dict = data['node_dict']
        edge_list = data['edge_list']
        return cls(node_dict, edge_list)
    
    def graph_coords_cvt(self, ct: Callable):
        for node_id, node_data in self.nodes(data=True):
            new_position = ct(node_data[self._position_key])
            self.nodes[node_id][self._position_key] = new_position
        self._distance_weight()

    def get_node_coord(self, node_id) -> tuple:
        x = self.nodes[node_id][self._position_key][0]
        y = self.nodes[node_id][self._position_key][1]
        return x, y

    def return_given_path(self, graph_node_ids: list) -> list[tuple]:
        return [self.get_node_coord(id) for id in graph_node_ids]

    def return_random_path(self, start_node_id, num_traversed_nodes:int) -> list[tuple]:
        """Return random GeometricGraphNode without repeat nodes
        """
        node_ids = [start_node_id]
        nodelist = [self.get_node_coord(start_node_id)]
        for _ in range(num_traversed_nodes):
            connected_node_ids = list(self.adj[node_ids[-1]])
            connected_node_ids = [x for x in connected_node_ids if x not in node_ids]
            if not connected_node_ids:
                return nodelist
            next_id = random.choice(connected_node_ids) # NOTE: Change this to get desired path pattern
            node_ids.append(next_id)
            nodelist.append(self.get_node_coord(next_id))
        return nodelist
    
    
    def plot(self, ax: Axes, node_style='x', node_text:bool=True, edge_color='r'):
        self.plot_graph(ax, node_style, node_text, edge_color)

    def plot_graph(self, ax: Axes, node_style='x', node_text:bool=True, edge_color='r'):
        if node_style is not None:
            self.plot_graph_nodes(ax, node_style, node_text)
        if edge_color is not None:
            self.plot_graph_edges(ax, edge_color)

    def plot_graph_nodes(self, ax: Axes, style='x', with_text=True):
        [ax.plot(self.get_node_coord(n)[0], self.get_node_coord(n)[1], style) for n in list(self.nodes)]
        if with_text:
            [ax.text(self.get_node_coord(n)[0], self.get_node_coord(n)[1], n) for n in list(self.nodes)]

    def plot_graph_edges(self, ax: Axes, edge_color='r'):
        nx.draw_networkx_edges(self, nx.get_node_attributes(self, self._position_key), ax=ax, edge_color=edge_color)




