import os
import json
import pathlib
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from basic_map.graph import NetGraph


def read_graph_json(json_path:str) -> NetGraph:
    with open(json_path, 'r') as jf:
        data = json.load(jf)
    node_dict = data['node_dict']
    edge_list = data['edge_list']
    return NetGraph(node_dict, edge_list)


project_root = pathlib.Path(__file__).resolve().parents[1]
map_png_path = os.path.join(project_root, 'data', 'warehouse_sim_original', 'graph.json')

graph = read_graph_json(map_png_path)

fig, ax = plt.subplots()
ax = cast(Axes, ax)

graph.plot_graph(ax)
ax.axis('equal')

plt.show()

