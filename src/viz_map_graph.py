import os
import json
import pathlib

import matplotlib.pyplot as plt

from basic_map.graph import NetGraph


def read_graph_json(json_path:str) -> NetGraph:
    with open(json_path, 'r') as jf:
        data = json.load(jf)
    node_dict = data['node_dict']
    edge_list = data['edge_list']
    return NetGraph(node_dict, edge_list)

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'warehouse_sim_original')

MAP_PATH = os.path.join(DATA_DIR, 'background.png')
GRAPH_PATH = os.path.join(DATA_DIR, 'graph.json')
GRAPH_JSON_PATH = os.path.join(DATA_DIR, 'graph.json')

graph = read_graph_json(GRAPH_JSON_PATH)

fig, ax = plt.subplots()
ax.imshow(plt.imread(MAP_PATH))
graph.plot_graph(ax, node_coord=True)
ax.axis('equal')

plt.show()