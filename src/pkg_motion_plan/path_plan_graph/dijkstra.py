import copy
from heapq import heappush, heappop
from typing import Any

import networkx as nx # type: ignore


class DijkstraPathPlanner:
    def __init__(self, graph:nx.Graph):
        """Find the shortest path between two nodes in a graph based on Dijkstra's algorithm."""
        self.G = graph

    def k_shortest_paths(self, source, target, k=1, weight_key='weight') -> tuple[list[float], list[list[Any]]]:
        """Returns the k-shortest paths from source to target in a weighted graph G.

        Args:
            source/target: Networkx node index
            k     : The number of shortest paths to find
            weight_key: Edge data key corresponding to the edge weight

        Returns:
            lengths: Stores the length of each k-shortest path.
            paths  : Stores each k-shortest path. 

        Raises:
            NetworkXNoPath: If no path exists between source and target.

        Notes:
            Edge weight attributes must be numerical and non-negative.
            Distances are calculated as sums of weighted edges traversed.

        References:
            Source code: 'Guilherme Maia <guilhermemm@gmail.com>'.
            Algorithm: 'An algorithm for finding the k quickest paths in a network' Y.L.Chen
        """
        if source == target:
            return ([0], [[source]]) 
        G = copy.deepcopy(self.G) # self.NG is the original graph
        
        length, path = nx.single_source_dijkstra(G, source, target, weight=weight_key)
        if target not in path:
            raise nx.NetworkXNoPath("Node %s not reachable from %s." % (source, target))
            
        lengths = [length] # init lengths
        paths = [path]     # init paths
        cnt = 0 
        B:list[tuple[float, int, Any]] = []   
        for _ in range(1, k):
            for j in range(len(paths[-1]) - 1):            
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]
                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G.has_edge(u, v):
                            edge_attr = G.edges[u,v]
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))
                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    for u, v, edge_attr in list(G.edges(node, data=True)):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
                try:
                    spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight_key)
                except:
                    continue
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_path_length = self.get_path_length(root_path, weight_key) + spur_path_length               
                    heappush(B, (total_path_length, cnt, total_path))
                    cnt += 1
                for e in edges_removed:
                    u, v, edge_attr = e
                    G.add_edge(u, v)
                    for key, value in edge_attr.items():
                        G[u][v][key] = value
                        
            if B:
                (l, _, p) = heappop(B)        
                lengths.append(l)
                paths.append(p)
            else:
                break
        
        return lengths, paths

    def k_shortest_paths_with_coords(self, source, target, k=1, weight_key='weight', position_key='position') -> tuple[list[float], list[list[tuple[float, float, Any]]]]:
        """Find the k shortest paths from source to target with coordinates.

        Args:
            source: Source node index.
            target: Target node index.
            k: The number of shortest paths. If k=1, get the shortest path.
            weight_key: Name tag of the weight. Default is "weight".
            position_key: Name tag of the position. Default is "position".

        Raises:
            KeyError: If the position_key is not in the node attributes.

        Returns:
            lengths: List of lengths of obtained paths.
            paths: List of obtained paths. Each path is a list of (x, y, node_id) tuples.
        """
        lengths, _paths = self.k_shortest_paths(source, target, k, weight_key)
        paths = []
        for _path in _paths:
            path = []
            for node_id in _path:
                x, y = self.G.nodes[node_id][position_key][:2]
                path.append((float(x), float(y), node_id))
        paths.append(path)
        return lengths, paths

    def get_path_length(self, path_node_idc: list, weight_key:str='weight') -> float:
        """Get the length of a path defined by a list of node indices.

        Args:
            path_node_idc: List of node indices.
            weight_key: Name of the weight, normally just "weight".

        Returns:
            length: Length of the path.
        """
        length = 0
        if len(path_node_idc) > 1:
            for i in range(len(path_node_idc) - 1):
                u, v = path_node_idc[i], path_node_idc[i+1]
                length += self.G.edges[u, v][weight_key]
        return length


if __name__ == "__main__":
    """
    Example graph:
        0 - 1 - 2
         \   \  |
                3
    """
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(1, 2, weight=1)
    graph.add_edge(2, 3, weight=2)
    graph.add_edge(1, 3, weight=2)
    graph.add_edge(0, 3, weight=3)
    dij_path_planner = DijkstraPathPlanner(graph)
    lengths, paths = dij_path_planner.k_shortest_paths(0, 3, k=2)
    print(lengths)
    print(paths)
